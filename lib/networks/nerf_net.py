import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.config import cfg
from . import embedder
import numpy as np
from .projection.map import SurfaceAlignedConverter
from .gconv import GCN
from .smpl_optimize import LearnableSMPL
import time
import psutil
from .unet_model import UNet
from lib.utils import net_utils, sdf_utils, sample_utils, blend_utils
import os

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.train_frame_start = cfg.begin_ith_frame
        self.train_frame_end = cfg.begin_ith_frame + cfg.num_train_frame
        # self.gcn = GCN()
        self.nerf = NeRF()
        # self.converter = SurfaceAlignedConverter()
        # self.learnable_smpl = LearnableSMPL(requires_grad=cfg.optimize_smpl)
        #
        self.pose_human = PoseHuman()



    def pts_to_can_pts(self, pts, sp_input):
        """transform pts from the world coordinate to the smpl coordinate"""
        Th = sp_input['Th']
        pts = pts - Th
        R = sp_input['R']
        pts = torch.matmul(pts, R)
        return pts

    # def get_mask(self, h, thres=0.15):
    #     mask = (h < thres) & (h > -0.5) # (batch, 65536)
    #     return mask.squeeze()

    # def get_mask(self, inp, thres=0.15):
    #     h = inp[..., -1]
    #     if inp.shape[-1] == 3:
    #         mask = torch.ones_like(h).bool()
    #     else:
    #         mask = (h < thres) & (h > -0.5) # (batch, 65536)
    #     return mask

    def prepare_input(self, wpts, batch):
        # frame_index = batch['frame_index']
        # if self.training:
        #     if cfg.optimize_smpl and (frame_index >= self.train_frame_start) and (frame_index < self.train_frame_end):
        #         frame = (frame_index - self.train_frame_start) // cfg.frame_interval
        #         verts = self.learnable_smpl.get_learnable_verts(frame)
        #     else:
        #         verts = batch['wverts']
        # else:
        #     if cfg.shape_control[0]:
        #         frame = (frame_index - self.train_frame_start) // cfg.frame_interval
        #         verts = self.learnable_smpl.get_learnable_verts(frame, cfg.shape_control)
        #     else:
        #         verts = batch['wverts']

        # transform sample points to surface-aligned representation (inp)
        # inp, local_coordinates, nearest_new = self.converter.xyz_to_xyzch(wpts, verts, batch=batch) # (65536, 7)
        # return inp, local_coordinates
        return wpts, None

    def forward(self, wpts, wdir, dists, batch):
        wdir = wdir # [1,n_rays*n_sample,3]

        ret = self.pose_human(wpts, wdir, dists, batch)

        # 使用bound mask
        # wbounds = batch['wbounds'][0].clone()
        # wbounds[0] -= 0.05
        # wbounds[1] += 0.05
        # inside = wpts > wbounds[:1]
        # inside = inside * (wpts < wbounds[1:])
        # outside = torch.sum(inside, dim=1) != 3
        # ret['raw'] = ret['raw'].transpose(1, 2).squeeze(0)
        # ret['raw'][outside] = 0

        # 恢复projection mask
        raw = ret['raw']
        ret.update({'raw': raw[None]})
        return ret


    def get_alpha(self, wpts, batch):
        inp, local_coordinates= self.prepare_input(wpts, batch)
        n_point = inp.shape[1]
        wpts = inp[..., :3] # [1,n_rays*n_sample,3]
        wproj = inp[..., 3:6] # [1,n_rays*n_sample,3]
        h = inp[..., 6, None] # [1,n_rays*n_sample,1]
        # use_proj_mask = True
        # if cfg.always_fill or (not cfg.use_proj_mask) or batch['mode'] != 'train':# todo：需要更新到别的对比实验里
        #     use_proj_mask = False
        # if use_proj_mask:
        #     # 加projection mask
        #     proj_mask = self.get_mask(h) # [n_rays*n_sample] remove bad projections
        #     batch['proj_mask'] = proj_mask
        #     if proj_mask.max() == 0:
        #         ret = {'raw': torch.zeros([1, n_point, 4], device=wpts.device),}
        #         return ret
        #     # with torch.no_grad():
        #     wpts = wpts[proj_mask]
        #     wproj = wproj[proj_mask]
        #     h = h[proj_mask]

        inputs = torch.cat([wpts, wproj, h], dim=-1)
        alpha = self.pose_human.get_alpha(inputs, batch)


        # 使用bound mask
        # wbounds = batch['wbounds'][0].clone()
        # wbounds[0] -= 0.05
        # wbounds[1] += 0.05
        # inside = wpts > wbounds[:1]
        # inside = inside * (wpts < wbounds[1:])
        # outside = torch.sum(inside, dim=1) != 3
        alpha = alpha.transpose(1, 2).squeeze(0)
        # alpha[outside] = 0
        # if use_proj_mask:
        #     # 恢复projection mask
        #     raw = torch.zeros([n_point, 4], device=wpts.device)
        #     raw[proj_mask] = alpha
        # else:
        raw = alpha
        # ret.update({'raw': raw[None]})

        return raw

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
        # self.res1 = ResBlock()
        # self.res2 = ResBlock()
        # self.res3 = ResBlock()
        # self.res4 = ResBlock()
        # self.res5 = ResBlock()
    def forward(self, x):
        x = F.relu(self.l0(x))
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = F.relu(self.l5(x))

        return x



class NeRF(nn.Module):
    def __init__(self):
        super(NeRF, self).__init__()

        self.alpha_l1 = nn.Conv1d(256, 1, 1)
        self.rgb_l1 = nn.Conv1d(259, 128, 1) # for shallow net
        self.rgb_l2 = nn.Conv1d(128, 3, 1)
        d_in = 0
        if cfg.sdf_input.wpts:
            d_in += 3 * 13
        self.mlp = CondMLP(input_dim=d_in)
        self.embed_fn_fine = None

        multires = 6
        if multires > 0:
            embed_fn, input_ch = embedder.get_embedder(multires,
                                                       input_dims=d_in)
            self.embed_fn_fine = embed_fn



    def get_alpha(self, inputs, batch):
        # x: xyzch
        # d: direction
        # z: pose
        n_points = inputs.shape[0]

        feat = inputs
        if self.embed_fn_fine is not None:
            feat = self.embed_fn_fine(feat)
        x = feat.permute(1, 0).unsqueeze(0)#[1,311,630]
        feat = self.mlp(x)
        # density
        alpha = self.alpha_l1(feat)#[32768,45]
        return alpha

    def forward(self, inputs, wldir, batch):
        # x: xyzch
        # d: direction
        # z: pose
        n_points = inputs.shape[0]

        feat = inputs
        if self.embed_fn_fine is not None:
            feat = self.embed_fn_fine(feat)
        x = feat.permute(1, 0).unsqueeze(0)#[1,311,630]
        feat = self.mlp(x)
        # density
        alpha = self.alpha_l1(feat)#[32768,45]
        # rgb
        # color_dir_input = net_utils.filter_color_dir_feature(wldir).permute(1, 0).unsqueeze(0)
        wdir = wldir[:, :3].permute(1, 0).unsqueeze(0)
        feat = torch.cat((feat, wdir), dim=1)
        rgb = F.relu(self.rgb_l1(feat))
        rgb = self.rgb_l2(rgb)
        return rgb, alpha


class PoseHuman(nn.Module):
    def __init__(self):
        super(PoseHuman, self).__init__()
        self.nerf = NeRF()
    def get_alpha(self, inputs, batch):
        nerf_input = net_utils.filter_sdf_feature(inputs)
        return self.nerf.get_alpha(nerf_input, batch)

    def forward(self, inputs, wldir, dists, batch):
        return self.nerf_forward(inputs, wldir, dists, batch)

    def nerf_forward(self, inputs, wldir, dists, batch):
        # nerf_input = net_utils.filter_sdf_feature(inputs)
        wpts = inputs[:, :3]
        rgb, alpha = self.nerf(wpts, wldir, batch)
        ret = {'raw' : torch.cat((rgb, alpha), dim=1)}
        return ret
########################################################################################################
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from lib.config import cfg
# from . import embedder
#
# from .projection.map import SurfaceAlignedConverter
# from .gconv import GCN
# from .smpl_optimize import LearnableSMPL
#
#
# class Network(nn.Module):
#     def __init__(self):
#         super(Network, self).__init__()
#
#         self.train_frame_start = cfg.begin_ith_frame
#         self.train_frame_end = cfg.begin_ith_frame + cfg.num_train_frame
#
#         # self.gcn = GCN()
#         self.nerf = NeRF()
#         # self.converter = SurfaceAlignedConverter()
#         # if cfg.vert_type == 'smpl':
#         #     self.leanable_smpl = LearnableSMPL(requires_grad=cfg.optimize_smpl)
#
#     def pts_to_can_pts(self, pts, sp_input):
#         """transform pts from the world coordinate to the smpl coordinate"""
#         Th = sp_input['Th']
#         pts = pts - Th
#         R = sp_input['R']
#         pts = torch.matmul(pts, R)
#         return pts
#
#     def get_mask(self, inp, thres=0.15):
#         h = inp[..., -1]
#         if inp.shape[-1] == 3:
#             mask = torch.ones_like(h).bool()
#         else:
#             mask = (h < thres) & (h > -0.5) # (batch, 65536)
#         return mask
#
#
#     def prepare_input(self, wpts, sp_input):
#
#         # frame_index = sp_input['frame_index']
#         # if self.training:
#         #     if cfg.optimize_smpl and (frame_index >= self.train_frame_start) and (frame_index < self.train_frame_end) and cfg.vert_type == 'smpl':
#         #         frame = (frame_index - self.train_frame_start) // cfg.frame_interval
#         #         verts = self.leanable_smpl.get_learnable_verts(frame)
#         #     else:
#         #         verts = sp_input['verts_world']
#         # else:
#         #     if cfg.shape_control[0] and cfg.vert_type == 'smpl':
#         #         frame = (frame_index - self.train_frame_start) // cfg.frame_interval
#         #         verts = self.leanable_smpl.get_learnable_verts(frame, cfg.shape_control)
#         #     else:
#         #         verts = sp_input['verts_world']
#
#         # transform sample points to surface-aligned representation (inp)
#         # inp, local_coordinates = self.converter.xyz_to_xyzch(wpts, verts) # (batch, 65536, 4)
#         return wpts, None
#         # return inp, local_coordinates
#
#
    # def forward(self, inp, viewdir, dists, sp_input):
    #     # mask points far away from mesh surface
    #     mask = self.get_mask(inp)
    #     inp_f, mask_f = inp.view(-1, inp.shape[-1]), mask.view(-1)
    #     # raw_all = torch.zeros(*mask_f.shape, 4, device=inp.device)
    #     # # if all points are far from body
    #     # if not torch.any(mask_f):
    #     #     return raw_all.view(*mask.shape, 4) # rgbd
    #     # inp_f_masked = inp_f[mask_f].unsqueeze(0) # (1, len(mask_f), 4)
    #     inp_f_masked = inp_f.unsqueeze(0) # (1, len(mask_f), 4)
    #     # positional encoding for inp
    #     inp_f_masked = embedder.inp_embedder(inp_f_masked).transpose(1, 2) # (1, 52, len(mask_f))
    #
    #     # get pose embedding
    #     # poses = sp_input['poses'] # (batch, 24, 3)
    #     # poses = poses.transpose(1, 2)
    #     # pose_embed = self.gcn(poses) # (batch, 64, 24)
    #     # pose_embed = torch.mean(pose_embed, dim=-1, keepdim=True) # (batch, 64, 1)
    #     # pose_embed_f = pose_embed.repeat(1, 1, inp_f_masked.shape[-1])
    #
    #     # get viewdir
    #     viewdir_f_masked = viewdir.view(-1, 3).unsqueeze(0).transpose(1, 2)
    #     # viewdir_f_masked = viewdir.view(-1, 3)[mask_f].unsqueeze(0).transpose(1, 2)
    #     # local_coordinates = local_coordinates[mask_f]
    #     # viewdir_local = torch.matmul(local_coordinates, viewdir_f_masked.unsqueeze(2)).squeeze(2) # 这里把view direction转换成了局部坐标系下的view direction
    #     # viewdir_f_masked = torch.cat((viewdir_f_masked, viewdir_local), dim=1).unsqueeze(0)
    #
    #     # positional encoding for viewdir
    #     # viewdir_f_masked = embedder.w_view_embedder(viewdir_f_masked).transpose(1, 2)
    #
    #     # forward
    #     rgb, alpha = self.nerf(inp_f_masked, viewdir_f_masked)
    #
    #     raw = torch.cat((rgb, alpha), dim=1)
    #     raw = raw.transpose(1, 2) # (1, len(mask_f), 4)
    #     raw_all = raw.squeeze(0)
    #     ret = {'raw':raw_all.view(*mask.shape, 4)}
    #     return ret
    #     # raw_all[mask_f] = raw.squeeze(0)
    #     #
    #     # return raw_all.view(*mask.shape, 4)
#
#
#     def calculate_density_color(self, wpts, viewdir, sp_input):
#
#         inp, local_coordinates = self.prepare_input(wpts, sp_input)
#         ret = self.forward(inp, viewdir, None, sp_input)
#         raw = ret['raw']
#         return raw
#
# class CondMLP(nn.Module):
#     def __init__(self, input_dim=39):
#         super(CondMLP, self).__init__()
#         # 52 = 4*13 no norm
#         # 52+3 no embedding
#         # 91 = 7*13 embedding norm
#
#         self.l0 = nn.Conv1d(input_dim, 256, 1)
#         self.l1 = nn.Conv1d(256, 256, 1)
#         self.l2 = nn.Conv1d(256, 256, 1)
#         self.l3 = nn.Conv1d(256, 256, 1)
#         self.l4 = nn.Conv1d(256, 256, 1)
#         self.l5 = nn.Conv1d(256, 256, 1)
#     def forward(self, x):
#         x = F.relu(self.l0(x))
#         x = F.relu(self.l1(x))
#         x = F.relu(self.l2(x))
#         x = F.relu(self.l3(x))
#         x = F.relu(self.l4(x))
#         x = F.relu(self.l5(x))
#
#         return x
#
#
#
# class NeRF(nn.Module):
#     def __init__(self):
#         super(NeRF, self).__init__()
#         self.mlp = CondMLP()
#         self.alpha_l1 = nn.Conv1d(256, 1, 1)
#         self.rgb_l1 = nn.Conv1d(259, 128, 1) # for shallow net
#         self.rgb_l2 = nn.Conv1d(128, 3, 1)
#
#     def forward(self, x, d):
#         feat = self.mlp(x)
#         # density
#         alpha = self.alpha_l1(feat)
#         # rgb
#         feat = torch.cat((feat, d), dim=1)
#         rgb = F.relu(self.rgb_l1(feat))
#         rgb = self.rgb_l2(rgb)
#         return rgb, alpha