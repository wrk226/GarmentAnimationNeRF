import math
from threading import local
import torch
from lib.config import cfg
from .. import embedder
import time
import psutil
from . import nerf_net_utils
from lib.networks.renderer import make_renderer
from lib.visualizers import make_visualizer
from lib.utils.ray_tracing import RayTracing

class Renderer:
    def __init__(self, net, net_upper_body=None, net_lower_body=None):
        self.ray_tracer = RayTracing(cfg.ray_tracer)
        self.net = net
        self.mesh_renderer = make_renderer(cfg.mesh, self.net)
        self.mesh_visualizer = make_visualizer(cfg.mesh)
        self.clothes_change = False
        if net_upper_body or net_lower_body:
            self.clothes_change = True
            self.net_upper = net_upper_body if net_upper_body is not None else net
            self.net_lower = net_lower_body if net_lower_body is not None else net

    def get_sampling_points(self, ray_o, ray_d, near, far):
        # calculate the steps for each ray
        t_vals = torch.linspace(0., 1., steps=cfg.N_samples, device=near.device)#.to(near)
        z_vals = near[..., None] * (1. - t_vals) + far[..., None] * t_vals

        if cfg.perturb > 0. and self.net.training:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape, device=upper.device)#.to(upper)
            z_vals = lower + (upper - lower) * t_rand

        pts = ray_o[:, :, None] + ray_d[:, :, None] * z_vals[..., None]


        return pts, z_vals


    def get_density_color(self, batch, sr_info=None):

        ret = {}

        nr_feat = self.net.prepare_input(batch, sr_info=sr_info)
        ret['nr_feat'] = nr_feat
        return ret

    def get_segment_mask(self, inp, thres_upper=0.32, thres_lower=-0.23):
        z = inp[..., 1]
        mask_head = (z >= thres_upper)
        mask_upper = (z < thres_upper) & (z > thres_lower)
        mask_lower = (z <= thres_lower)
        return mask_head, mask_upper, mask_lower

    def get_upsampling_points(self, ray_o, ray_d, z_vals, weights, batch):
        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = nerf_net_utils.sample_pdf(z_vals_mid, weights[...,1:-1], cfg.N_importance, det=(cfg.perturb==0.))
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = ray_o[...,None,:] + ray_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]
        return pts[None], z_vals[None]


    def get_pixel_value(self, ray_o, ray_d, near, far, occ, batch):
        # surface_rendering
        if cfg.rotate_camera:
            body_img, body_depth_map, _ = self.net.surface_renderer.render_with_light(batch['sr_sverts'],batch['sr_faces'].long(), batch['frame_index'],
                                                                                   batch['cam_ind'],
                                                                                   extra_rot=batch['extra_rot'],
                                                                                   mode=batch['mode'],
                                                                                      height=cfg.unet_dim, width=cfg.unet_dim
                                                                                   )
            _, _, sr_info = self.net.surface_renderer.render_with_light(batch['wverts'],batch['faces'].long(), batch['frame_index'],
                                                                                   batch['cam_ind'],
                                                                                   extra_rot=batch['extra_rot'],
                                                                                   mode=batch['mode'],
                                                                        height=cfg.unet_dim, width=cfg.unet_dim
                                                                                   )
        else:
            body_img, body_depth_map, _ = self.net.surface_renderer.render_with_light(batch['sr_sverts'],batch['sr_faces'].long(), batch['frame_index'],
                                                                                   batch['cam_ind'],
                                                                                   mode=batch['mode'],
                                                                                      height=cfg.unet_dim, width=cfg.unet_dim
                                                                                   )
            _, _, sr_info = self.net.surface_renderer.render_with_light(batch['wverts'],batch['faces'].long(), batch['frame_index'],
                                                                                   batch['cam_ind'],
                                                                                   mode=batch['mode'],
                                                                        height=cfg.unet_dim, width=cfg.unet_dim
                                                                                   )
        sr_info['body_rgb'] = body_img#[batch['select_coord'][0,chunk_s:chunk_e, 0], batch['select_coord'][0,chunk_s:chunk_e, 1]]
        # sampling points along camera rays
        # [1, 1024, 64, 3]
        wpts, z_vals = self.get_sampling_points(ray_o, ray_d, near, far)


        # compute the color and density
        ret = self.get_density_color(batch, sr_info=sr_info)
        n_batch, n_pixel, n_sample = z_vals.shape


        feat_map = ret['nr_feat'].permute(2,3,1,0).squeeze()

        sr_body_img = body_img
        feat_map = torch.cat([feat_map, sr_body_img], dim=2)

        rgb_nr_map = self.net.neural_renderer(feat_map.permute(2,0,1).unsqueeze(0)).permute(0,2,3,1).squeeze()
        n_channel = rgb_nr_map.shape[-1]
        rgb_map = rgb_nr_map.reshape(n_batch, -1, n_channel)



        ret['rgb_map'] = rgb_map


        if batch['mode'] == 'train':
            output = ret
        else:
            output = {
                'rgb_map': rgb_map,
            }

        if not rgb_map.requires_grad:
            output = {k: output[k].detach() for k in output.keys()}

        return output

    def get_neighbours(self, alpha_voxel):
        n_channel = alpha_voxel.shape[1]
        import torch.nn.functional as F
        # 定义6个卷积核
        up_kernel = torch.zeros((n_channel, n_channel, 3, 3, 3), dtype=torch.float32, device='cuda')
        for i in range(n_channel):
            up_kernel[i,i,0,1,1] = 1
        down_kernel = torch.zeros((n_channel, n_channel, 3, 3, 3), dtype=torch.float32, device='cuda')
        for i in range(n_channel):
            down_kernel[i,i,2,1,1] = 1
        left_kernel = torch.zeros((n_channel, n_channel, 3, 3, 3), dtype=torch.float32, device='cuda')
        for i in range(n_channel):
            left_kernel[i,i,1,0,1] = 1
        right_kernel = torch.zeros((n_channel, n_channel, 3, 3, 3), dtype=torch.float32, device='cuda')
        for i in range(n_channel):
            right_kernel[i,i,1,2,1] = 1
        front_kernel = torch.zeros((n_channel, n_channel, 3, 3, 3), dtype=torch.float32, device='cuda')
        for i in range(n_channel):
            front_kernel[i,i,1,1,0] = 1
        back_kernel = torch.zeros((n_channel, n_channel, 3, 3, 3), dtype=torch.float32, device='cuda')
        for i in range(n_channel):
            back_kernel[i,i,1,1,2] = 1
        # alpha_voxel=1
        # 对每个方向进行卷积操作
        up = F.conv3d(alpha_voxel, up_kernel, padding=(1,1,1))
        down = F.conv3d(alpha_voxel, down_kernel, padding=(1,1,1))
        left = F.conv3d(alpha_voxel, left_kernel, padding=(1,1,1))
        right = F.conv3d(alpha_voxel, right_kernel, padding=(1,1,1))
        front = F.conv3d(alpha_voxel, front_kernel, padding=(1,1,1))
        back = F.conv3d(alpha_voxel, back_kernel, padding=(1,1,1))

        return torch.cat([up, down, left, right, front, back], dim=0)

    def render(self, batch):
        ray_o = batch['ray_o']
        ray_d = batch['ray_d']
        near = batch['near']
        far = batch['far']
        # if not cfg.neural_rendering:
        # if batch['mode'] == 'train':
        #     occ = batch['occupancy']
        
        # volume rendering for each pixel
        n_batch, n_pixel = ray_o.shape[:2]
        chunk = cfg.chunk
        ret_list = []
        for i in range(0, n_pixel, chunk):
            ray_o_chunk = ray_o[:, i:i + chunk]
            ray_d_chunk = ray_d[:, i:i + chunk]
            near_chunk = near[:, i:i + chunk]
            far_chunk = far[:, i:i + chunk]
            batch['curr_chunk'] = [i, i + chunk]

            # if cfg.neural_rendering:
            #     occ_chunk = None
            # else:
            # if batch['mode'] == 'train':
            #     occ_chunk = occ[:, i:i + chunk]

            pixel_value = self.get_pixel_value(ray_o_chunk, ray_d_chunk,
                                               near_chunk, far_chunk,
                                               None, batch)
            if batch['mode'] != 'train':
                pixel_value = {k: pixel_value[k].detach().cpu() for k in pixel_value.keys()}
            ret_list.append(pixel_value)

        keys = ret_list[0].keys()
        ret = {k: torch.cat([r[k] for r in ret_list], dim=1) for k in keys}

        # if batch['mode'] != 'train' and not cfg.only_render_body and cfg.neural_rendering:
        #     nr_rgb_pred = ret['rgb_map'][..., :3]
        #     nr_mask_pred = ret['rgb_map'][..., -1, None]
        #     if cfg.surface_rendering and cfg.surface_rendering_v3:
        #         sr_body_rgb = batch['body_rgb'].cpu()
        #         ret['rgb_map'] = nr_rgb_pred * nr_mask_pred + (1-nr_mask_pred)*sr_body_rgb
        #     else:
        #         ret['rgb_map'] = nr_rgb_pred * nr_mask_pred
        return ret

    def get_mesh(self, batch,get_alpha=False):
        for k in batch:
            if torch.is_tensor(batch[k]):#k != 'meta' and k != 'iter_step':
                batch[k] = batch[k].cuda()
        with torch.no_grad():
            output = self.mesh_renderer.render(batch)#,get_alpha=get_alpha)
            self.mesh_visualizer.visualize(output, batch)

    def get_smpl_mesh(self, batch):
        with torch.no_grad():
            # faces = self.net.learnable_smpl.body_model.faces
            faces = batch['faces'][0].cpu().numpy()
            input = {'triangle': faces,}
            if hasattr(self.net, 'curr_smpl_wverts'):
                input['posed_vertex'] = self.net.curr_smpl_wverts[0].cpu().numpy()
            else:
                input['posed_vertex'] = batch['wverts'][0].cpu().numpy()
            self.mesh_visualizer.visualize(input, batch, is_smpl=True)