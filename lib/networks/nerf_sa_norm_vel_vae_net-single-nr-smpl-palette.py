import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.config import cfg
from . import embedder
import numpy as np
from .projection.map import SurfaceAlignedConverter
from .gconv import GCN
# from .smpl_optimize import LearnableSMPL
import time
import psutil
# from .unet_model import UNet
from .vae_block import AutoEncoder
from lib.utils import net_utils, sdf_utils, sample_utils, blend_utils, geom_utils
import os
from lib.utils import vis_utils
import cv2
from lib.networks.renderer.surface_renderer import SurfaceRenderer
from lib.networks.siren import GeoSIREN
from lib.networks.neural_renderer import NeuralRenderer, NeuralRenderer_palette
from lib.utils import surface_render_utils
from pytorch3d.renderer.cameras import PerspectiveCameras
# import spconv.pytorch as spconv
from lib.networks.sparseconv import SparseConvNet_64

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.c = nn.Embedding(cfg.sp_feat_dim, 16)
        self.xyzc_net = SparseConvNet_64()
        self.train_frame_start = cfg.begin_ith_frame - cfg.prev_frames
        self.train_frame_end = cfg.begin_ith_frame + cfg.num_train_frame
        # self.gcn = GCN()
        # self.nerf = NeRF()
        self.converter = SurfaceAlignedConverter()
        if cfg.descripter_encoder_type != '':
            self.unet = AutoEncoder(3*cfg.prev_frames*cfg.sdf_input.vel+3, cfg.descripter_dim,
                                    latent_dim=cfg.descripter_latent_dim, ae_type=cfg.descripter_encoder_type,
                                    use_flatten=cfg.down_sampling_layers==6 and cfg.descripter_decoder_flatten,
                                    n_down=cfg.down_sampling_layers,
                                    skip_connection=cfg.unet_skip_connection)
            self.unet_2d = AutoEncoder(3, cfg.descripter_dim,
                                        latent_dim=cfg.descripter_latent_dim, ae_type=cfg.descripter_encoder_type,
                                        use_flatten=cfg.down_sampling_layers==6 and cfg.descripter_decoder_flatten,
                                        n_down=cfg.down_sampling_layers,
                                        skip_connection=cfg.unet_skip_connection)
        self.pose_human = PoseHuman()
        self.surface_renderer = SurfaceRenderer(cfg.train_dataset.ann_file)
        cond_dim = 1
        self.mlp_tex = GeoSIREN(input_dim=3, z_dim=cond_dim, hidden_dim=128, output_dim=3, device='cuda', last_op=torch.sigmoid, scale=1).cuda()

        if cfg.neural_rendering:
            assert cfg.num_basis>0
            input_dim = 0

            if cfg.palette_input.sr_body:
                input_dim += 1
            out_dim = cfg.num_basis*(3+1) + cfg.radiance_dim + 1 + 3


            self.neural_renderer_palette_diffuse = NeuralRenderer_palette(img_size=512, upsample=False,nr_upsample_steps=cfg.nr_upsample_steps,
                                              input_dim=128 + input_dim, out_dim=out_dim, final_actvn=False)
            self.input_dim = input_dim
        if len(cfg.cloth_color)>0:
            basis_color = torch.tensor([cfg.cloth_color,
                                        cfg.body_color])
        else:
            basis_color = torch.tensor(cfg.palette_color)

        self.register_parameter('basis_color', nn.Parameter(basis_color, requires_grad=cfg.optimize_palette))

        self.basis_color_ori = basis_color.cuda().detach().clone()


    def pts_to_can_pts(self, pts, sp_input):
        """transform pts from the world coordinate to the smpl coordinate"""
        Th = sp_input['Th']
        pts = pts - Th
        R = sp_input['R']
        pts = torch.matmul(pts, R)
        return pts

    def get_mask(self, h, thres=0.15):
        mask = (h < thres) & (h > -0.5) # (batch, 65536)
        return mask.squeeze()

    def set_pytorch3d_intrinsic_matrix(self, K, H, W):
        fx = -K[0, 0] * 2.0 / W
        fy = -K[1, 1] * 2.0 / H
        px = -(K[0, 2] - W / 2.0) * 2.0 / W
        py = -(K[1, 2] - H / 2.0) * 2.0 / H
        K = [
            [fx, 0, px, 0],
            [0, fy, py, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ]
        K = np.array(K)
        return K

    def get_coord_for_each_spts(self, spts, batch, res):
        coords_lst = []
        resolution = res
        for view in range(len(cfg.reference_view)):

            R = batch['ref_R'][view][0]
            T = batch['ref_T'][view][0]
            K = batch['ref_K'][view][0].cpu()
            K[:2] = K[:2] * cfg.ratio
            pytorch3d_K = self.set_pytorch3d_intrinsic_matrix(K, resolution, resolution)

            cameras = PerspectiveCameras(device='cuda',
                                         K=pytorch3d_K[None],
                                         R=R.T[None],
                                         T=T.T)
            # transform vertices according to camera
            verts_view = cameras.get_world_to_view_transform().transform_points(spts)
            to_ndc_transform = cameras.get_ndc_camera_transform()
            verts_proj = cameras.transform_points(spts)
            verts_ndc = to_ndc_transform.transform_points(verts_proj)
            verts_ndc[..., 2] = verts_view[..., 2]
            verts_ndc = verts_ndc.clone()
            coords = torch.clip(1-((verts_ndc[..., :2]+1)/2)[...,[1,0]], 0, 1)
            coords_lst.append(coords)

        return coords_lst


    def prepare_sp_input(self, ppts, pverts, batch):
        # for spconv
        min_xyz = torch.min(ppts, dim=0).values
        max_xyz = torch.max(ppts, dim=0).values
        min_xyz -= cfg.box_padding
        max_xyz += cfg.box_padding
        # voxel_size = [0.002, 0.002, 0.002]
        voxel_size = [0.01, 0.01, 0.01]
        dhw = pverts[0, :, [2, 1, 0]]# xyz - > zyx = dhw
        min_dhw = min_xyz[[2, 1, 0]]
        max_dhw = max_xyz[[2, 1, 0]]
        voxel_size = torch.tensor(voxel_size, device='cuda')
        coord = torch.round((dhw - min_dhw) / voxel_size)

        # construct the output shape
        out_sh = torch.ceil((max_dhw - min_dhw) / voxel_size).int()
        x = 32
        out_sh = (out_sh | (x - 1)) + 1

        sp_input = {'coord': coord, 'out_sh': out_sh, 'batch_size': 1}

        feature_volume = self.encode_sparse_voxels(sp_input)
        # interpolate features
        dhw2 = ppts[..., [2, 1, 0]]
        dhw2 = dhw2 - min_dhw[None, :]
        dhw2 = dhw2 / voxel_size
        dhw2 = dhw2 / out_sh * 2 - 1 # 将dhw2归一化到[-1,1]
        # convert dhw to whd, since the occupancy is indexed by dhw
        grid_coords = dhw2[..., [2, 1, 0]]
        xyzc_features = self.interpolate_features(grid_coords.clone()[None,None,None], feature_volume)
        inp = xyzc_features.permute(2,1,0).squeeze()
        return inp

    def interpolate_features(self, grid_coords, feature_volume):
        features = []
        for volume in feature_volume:
            feature = F.grid_sample(volume,
                                    grid_coords,
                                    padding_mode='zeros',
                                    align_corners=True)
            features.append(feature)
        features = torch.cat(features, dim=1)
        features = features.view(features.size(0), -1, features.size(4))
        return features




    def get_reference_weight(self, cam_idx, batch):
        T = torch.from_numpy(np.array(self.surface_renderer.cameras['T'][cam_idx]))[None,:,0].cuda()/1000
        R = torch.from_numpy(np.array(self.surface_renderer.cameras['R'][cam_idx])).cuda()
        camera_location = - R.T @ T[0]
        ref_wt_uv = self.converter.get_ref_weights(batch['wverts'], batch['faces'].long(), camera_location)[None, None]
        return ref_wt_uv

    def prepare_input(self, spts, batch, n_pixel=None):
        s = time.time()
        ret = {}
        # prepare 3d feature
        # get projection on body
        body_proj_info = self.converter.xyz_to_xyzch(spts, batch['sverts'], batch=batch)
        for k in body_proj_info.keys():
            ret[f'body_{k}'] = body_proj_info[k]

        # get body descriptor
        if cfg.sdf_input.body_normal:

            if cfg.sdf_input.body_velocity:
                normal_uv = self.converter.extract_verts_normal_info(batch['pverts'])
                verts_prev = []
                vel_prev = []

                for i in range(cfg.prev_frames):
                    verts_prev.append(batch[f'sverts_prev_{i + 1}'])
                    vel_prev.append(batch[f'cano_vel_prev_{i + 1}'])

                velocity_uv = self.converter.extract_verts_velocity_info_v2(vel_prev)
                body_descriptor_uv = torch.cat([normal_uv, velocity_uv], dim=1)
            else:
                normal_uv = self.converter.extract_verts_normal_info(batch['pverts'])
                body_descriptor_uv = normal_uv

            if cfg.encode_body_descriptor:
                body_descriptor_uv = self.unet(body_descriptor_uv)
            ret['body_descriptor'] = self.converter.get_descriptor_for_each_point(body_descriptor_uv, body_proj_info, batch)

        if 'Th' in batch:
            ppts = (spts - batch['Th'][0])@ batch['R'][0]
        else:
            ppts = geom_utils.convert_s_to_p(spts, batch['wtrans'][0], batch['ptrans'][0], batch['prot'][0])

        # prepare 2d feature
        if cfg.sdf_input.rgb_gt:
            if cfg.n_reference_view == 2:
                if cfg.encode_2d_feat:
                    embedded_img = self.unet_2d(torch.cat(batch['ref_img_2d'], dim=0).permute(0,3,1,2))
                    ref_front_img = embedded_img[0].permute(1,2,0).contiguous()
                    ref_back_img = embedded_img[1].permute(1,2,0).contiguous()

                else:
                    ref_front_img = batch['ref_img_2d'][0][0]
                    ref_back_img = batch['ref_img_2d'][1][0]

                img_res = ref_front_img.shape[0]-1
                coords_lst = self.get_coord_for_each_spts(spts, batch, res=img_res + 1)
                # 用插值
                front_coord = (coords_lst[0]*img_res).long().contiguous()
                back_coord = (coords_lst[1]*img_res).long().contiguous()
                front_feat = ref_front_img[front_coord[:,0], front_coord[:,1]]
                back_feat = ref_back_img[back_coord[:,0], back_coord[:,1]]

                pix2face = self.surface_renderer.get_visiable_faces(batch['sverts'].clone(), batch['faces'].clone(), cam_id=cfg.reference_view[0]) # visiable from front(cam0)
                visiable_faces = set(torch.unique(pix2face.reshape(-1))[1:].cpu().numpy())
                proj_face_idx = body_proj_info['proj_face_idx'].squeeze().cpu().numpy()


                if cfg.ref_interp:
                    front_wt_uv = self.get_reference_weight(cfg.reference_view[0], batch)
                    back_wt_uv = self.get_reference_weight(cfg.reference_view[1], batch)

                    ref_wt = (self.converter.get_descriptor_for_each_point(torch.cat([front_wt_uv,back_wt_uv], dim=1), body_proj_info, batch)+1)/2
                    ref_wt = torch.nn.functional.normalize(ref_wt, p=2, dim=1)
                    ref_feats = torch.stack([front_feat, back_feat], dim=0)
                    ret['2d_feature'] = torch.sum(ref_feats.permute(1,0,2)*ref_wt[...,None],dim=1)
                else:
                    front_msk = torch.tensor([i in visiable_faces  for i in proj_face_idx], device='cuda')
                    select_2d_feature = torch.zeros(proj_face_idx.shape[0],ref_front_img.shape[-1], device='cuda')
                    select_2d_feature[front_msk] = front_feat[front_msk]
                    select_2d_feature[~front_msk] = back_feat[~front_msk]
                    ret['2d_feature'] = select_2d_feature
            else:
                raise NotImplementedError
        ret['ppts'] = ppts


        return ret

    def forward(self, inp, wdir, dists, batch):
        n_point = dists.shape[0]
        ppts = inp['ppts'] # [1,n_rays*n_sample,3]

        poses = batch['pjoints']#[1,72]


        # centerize the sample points
        nerf_inp = {'ppts': ppts,
                    'body_proj': inp['body_cano_proj'],
                    'coarse_proj': inp['coarse_garment_cano_proj'] if not cfg.body_only else None,
                    'body_h': inp['body_h'][..., None],
                    'coarse_h': inp['coarse_garment_h'][..., None] if not cfg.body_only else None,
                    'sp_feat': inp['sp_feat'] if cfg.sdf_input.spconv_feat else None,
                    '2d_feat': inp['2d_feature'] if cfg.sdf_input.rgb_gt else None,
                    'body_descriptor': inp['body_descriptor'] if cfg.sdf_input.body_normal else None,
                    'coarse_descriptor': inp['coarse_garment_descriptor'] if cfg.sdf_input.coarse_garment_normal else None,
                    'wdir': wdir,
                    'poses': poses,}


        ret = self.pose_human(nerf_inp, batch)

        # 使用bound mask
        pbounds = batch['pbounds'][0].clone()
        pbounds[0] -= 0.05
        pbounds[1] += 0.05
        inside = ppts > pbounds[:1]
        inside = inside * (ppts < pbounds[1:])
        outside = torch.sum(inside, dim=1) != 3
        ret['raw'] = ret['raw'].transpose(1, 2).squeeze(0)
        ret['raw'][outside] = 0

        raw = ret['raw']
        ret.update({'raw': raw[None]})
        return ret


class CondMLP(nn.Module):
    def __init__(self, input_dim=45):
        super(CondMLP, self).__init__()
        # 52 = 4*13 no norm
        # 52+3 no embedding
        # 91 = 7*13 embedding norm

        self.l0 = nn.Conv1d(input_dim, 256, 1)
        self.l1 = nn.Conv1d(256, 256, 1)
        self.l2 = nn.Conv1d(256, 256, 1)
        self.l3 = nn.Conv1d(256, 256, 1)
        self.l4 = nn.Conv1d(256, 256, 1)
        self.l5 = nn.Conv1d(256, 256, 1)


    def forward(self, x):
        x = F.relu(self.l0(x))
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = F.relu(self.l5(x))

        return x

class CondMLP_res(nn.Module):
    def __init__(self, input_dim=45):
        super(CondMLP_res, self).__init__()
        # 52 = 4*13 no norm
        # 52+3 no embedding
        # 91 = 7*13 embedding norm

        self.l0 = nn.Conv1d(input_dim, 256, 1)
        self.res1 = ResBlock()
        self.res2 = ResBlock()
        self.res3 = ResBlock()
        self.res4 = ResBlock()
        self.res5 = ResBlock()

    def forward(self, x):
        x = self.l0(x)
        z = x.clone()
        x = self.res1(x, z)
        x = self.res2(x, z)
        x = self.res3(x, z)
        x = self.res4(x, z)
        x = self.res5(x, z)

        return x

class CondMLP_resx2(nn.Module):
    def __init__(self, input_dim=45):
        super(CondMLP_resx2, self).__init__()
        # 52 = 4*13 no norm
        # 52+3 no embedding
        # 91 = 7*13 embedding norm

        self.l0 = nn.Conv1d(input_dim, 256, 1)
        self.res1 = ResBlock()
        self.res2 = ResBlock()
        self.res3 = ResBlock()
        self.res4 = ResBlock()
        self.res5 = ResBlock()
        self.res6 = ResBlock()
        self.res7 = ResBlock()
        self.res8 = ResBlock()
        self.res9 = ResBlock()
        self.res10 = ResBlock()

    def forward(self, x):
        x = self.l0(x)
        z = x.clone()
        x = self.res1(x, z)
        x = self.res2(x, z)
        x = self.res3(x, z)
        x = self.res4(x, z)
        x = self.res5(x, z)
        x = self.res6(x, z)
        x = self.res7(x, z)
        x = self.res8(x, z)
        x = self.res9(x, z)
        x = self.res10(x, z)
        return x


class ResBlock(nn.Module):
    def __init__(self):
        super(ResBlock, self).__init__()
        self.lz = nn.Conv1d(256, 256, 1)
        self.l1 = nn.Conv1d(256, 256, 1)
        self.l2 = nn.Conv1d(256, 256, 1)

    def forward(self, x, z):
        z = F.relu(self.lz(z))
        res = x + z
        x = F.relu(self.l1(res))
        x = F.relu(self.l2(x)) + res
        return x


class NeRF(nn.Module):
    def __init__(self):
        super(NeRF, self).__init__()

        self.alpha_l1 = nn.Conv1d(256, 1, 1)
        self.rgb_l1 = nn.Conv1d(259, 128, 1) # for shallow net
        self.rgb_l2 = nn.Conv1d(128, 3, 1)
        self.diff_rgb_l1 = nn.Conv1d(256, 128, 1)
        self.diff_rgb_l2 = nn.Conv1d(128, 3, 1)
        d_in = 0
        if cfg.sdf_input.t:
            self.color_latent = nn.Embedding(cfg.num_latent_code, 64)
            d_in += 64
        if cfg.sdf_input.wpts:
            if cfg.position_embedding:
                d_in += 3 * 13
            else:
                d_in += 3
        if cfg.sdf_input.wproj:
            if cfg.body_only or cfg.coarse_only:
                proj_dim = 3
            else:
                proj_dim = 6
            if cfg.position_embedding_proj:
                d_in += proj_dim * 13
            else:
                d_in += proj_dim

        if cfg.sdf_input.h:
            if cfg.body_only or cfg.coarse_only:
                d_in += 1
            else:
                d_in += 2

        if cfg.sdf_input.pose:
            d_in += 256
        if cfg.sdf_input.body_normal :
            if cfg.encode_body_descriptor:
                d_in += cfg.descripter_dim
            else:
                d_in += 3
        if cfg.sdf_input.coarse_garment_normal:
            if cfg.encode_coarse_garment_descriptor:
                d_in += cfg.descripter_dim
            else:
                d_in += 3
        if cfg.sdf_input.rgb_gt:

            if cfg.encode_2d_feat:
                if cfg.ref_interp:
                    d_in += cfg.descripter_dim
                else:
                    d_in += cfg.descripter_dim * cfg.n_reference_view//2
            else:
                d_in += 3 * cfg.n_reference_view//2
        if cfg.sdf_input.spconv_feat:
            d_in += 64


        if cfg.sdf_input.frame_idx:
            d_in += 13
        self.mlp = CondMLP(input_dim=d_in)

        multires = 6
        self.embed_fn_fine = None


        if multires > 0:
            embed_fn, input_ch = embedder.get_embedder(multires,
                                                       input_dims=d_in)
            self.embed_fn_fine = embed_fn
        self.gcn = GCN()

    def embed_poses(self, poses, n_points):
        poses = poses.transpose(1, 2)
        pose_embed = self.gcn(poses) # (batch, 64, 24)
        pose_embed = torch.mean(pose_embed, dim=-1, keepdim=True) # (batch, 64, 1)
        pose_embed = pose_embed.repeat(1, 1, n_points)# (1, 256, 21814)
        return pose_embed



    def forward(self, inputs, batch):
        # x: xyzch
        # d: direction
        # z: pose
        n_points = inputs['ppts'].shape[0]
        feat = None
        if cfg.sdf_input.wpts:
            feat = inputs['ppts']
            if cfg.position_embedding:
                feat = self.embed_fn_fine(feat)
        if cfg.sdf_input.wproj:
            if cfg.position_embedding_proj:
                if cfg.body_only:
                    proj_emb = self.embed_fn_fine(inputs['body_proj'])
                elif cfg.coarse_only:
                    proj_emb = self.embed_fn_fine(inputs['coarse_proj'])
                else:
                    proj_emb = self.embed_fn_fine(torch.cat([inputs['body_proj'], inputs['coarse_proj']], dim=-1))
                feat = proj_emb if feat is None else torch.cat([feat, proj_emb], dim=-1)
            else:
                if cfg.body_only:
                    proj_ori = inputs['body_proj']
                elif cfg.coarse_only:
                    proj_ori = inputs['coarse_proj']
                else:
                    proj_ori = torch.cat([inputs['body_proj'], inputs['coarse_proj']], dim=-1)
                feat = proj_ori if feat is None else torch.cat([feat, proj_ori], dim=-1)


        if cfg.sdf_input.h:
            if cfg.body_only:
                feat = torch.cat([feat, inputs['body_h']], dim=-1)
            elif cfg.coarse_only:
                feat = torch.cat([feat, inputs['coarse_h']], dim=-1)
            else:
                feat = torch.cat([feat, inputs['body_h'], inputs['coarse_h']], dim=-1)
        if cfg.sdf_input.pose:
            pose_embed = self.embed_poses(inputs['poses'], n_points).squeeze(0).permute(1, 0)#[12771, 256]
            # feat = self.mlp(inputs, pose_embed).squeeze(0).permute(1, 0)#[630,256]
            feat = torch.cat([feat, pose_embed], dim=-1)#[630,311]
        if cfg.sdf_input.spconv_feat:
            feat = torch.cat([feat, inputs['sp_feat']], dim=-1)#[630,311]
        if cfg.sdf_input.rgb_gt:
            feat = torch.cat([feat, inputs['2d_feat']], dim=-1)#[630,311]
        if cfg.sdf_input.body_normal and not cfg.coarse_only:
            feat = torch.cat([feat, inputs['body_descriptor']], dim=-1)
        if cfg.sdf_input.coarse_garment_normal and not cfg.body_only:
            feat = torch.cat([feat, inputs['coarse_descriptor']], dim=-1)
        if cfg.sdf_input.t:
            latent = self.color_latent(batch['latent_index'])
            latent = latent.expand(feat.size(0), latent.size(1))
            feat = torch.cat((feat, latent), dim=-1)
        x = feat.permute(1, 0).unsqueeze(0)#[1,311,630]
        feat = self.mlp(x)
        # density
        alpha = self.alpha_l1(feat)#[32768,45]
        # Diffuse rgb
        feat_d = feat
        feat_d = self.diff_rgb_l1(feat_d)
        # diff_rgb = F.sigmoid(self.diff_rgb_l2(F.relu(feat_d)))
        # view-dependent rgb
        feat_vd = torch.cat((feat, inputs['wdir'][None].permute(0,2,1)), dim=1)
        feat_vd = self.rgb_l1(feat_vd)
        # rgb = torch.sigmoid(self.rgb_l2(F.relu(feat_vd)))
        ret = {'raw': torch.cat([feat_d, feat_vd, alpha],dim=1)}
        return ret


class PoseHuman(nn.Module):
    def __init__(self):
        super(PoseHuman, self).__init__()
        self.nerf = NeRF()

    def forward(self, inp, batch):
        ret = self.nerf(inp, batch)
        return ret


