
import torch
from lib.config import cfg
import time
from . import nerf_net_utils
import torch.nn.functional as F

class Renderer:
    def __init__(self, net, net_upper_body=None, net_lower_body=None):
        self.net = net


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


    def get_density_color(self, wpts, viewdir, z_vals, batch, sr_info=None):
        """
        wpts: n_batch, n_pixel, n_sample, 3
        viewdir: n_batch, n_pixel, 3
        z_vals: n_batch, n_pixel, n_sample
        """
        n_batch, n_pixel, n_sample = wpts.shape[:3]
        wpts = wpts.view(n_batch * n_pixel * n_sample, -1)
        viewdir = viewdir[:, :, None].repeat(1, 1, n_sample, 1).contiguous()
        viewdir = viewdir.view(n_batch * n_pixel * n_sample, -1)

        # calculate dists for the opacity computation
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, dists[..., -1:]], dim=2)
        dists = dists.view(n_batch * n_pixel * n_sample)


        ret = {}
        if sr_info is None:
            inp= self.net.prepare_input(wpts, batch, n_pixel=n_pixel)
            ret = self.net(inp, viewdir, dists, batch)
        else:
            inp, local_coordinates, nr_feat = self.net.prepare_input(wpts, batch, n_pixel=n_pixel, sr_info=sr_info)
            if not cfg.nr_only:
                ret = self.net(inp, local_coordinates, viewdir, dists, batch)
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

        sr_info = None

        # sampling points along camera rays
        # [1, 1024, 64, 3]
        wpts, z_vals = self.get_sampling_points(ray_o, ray_d, near, far)
        # viewing direction
        viewdir = ray_d
        # compute the color and density
        ret = self.get_density_color(wpts, viewdir, z_vals, batch, sr_info=sr_info)
        n_batch, n_pixel, n_sample = z_vals.shape
        # volume rendering for wpts
        res = torch.sqrt(torch.tensor(cfg.N_rand)).int()
        raw_dim = 128+128+1
        raw = ret['raw'].reshape(-1, n_sample, raw_dim)# rgb+alpha

        raw_pixel, disp_map, acc_map, weights, depth_map = nerf_net_utils.raw2outputs_neus(
            raw, z_vals.view(-1, n_sample), cfg.white_bkgd)

        feat_d_pixel = raw_pixel[..., :128]
        feat_d_map = torch.zeros(res,res,128, device='cuda')
        feat_d_map[batch['select_coord'][...,0],batch['select_coord'][...,1]] = feat_d_pixel
        feat_d_map = feat_d_map.view(res,res,-1)

        if 'body_rgb' in batch:
            sr_body_img = batch['body_rgb'].reshape(512,512,3)[::512//res,::512//res][...,[0]]
            feat_d_map = torch.cat([feat_d_map, sr_body_img], dim=2)
        raw_diffuse_nr_map = self.net.neural_renderer_palette_diffuse(feat_d_map.permute(2,0,1).unsqueeze(0)).permute(0,2,3,1).squeeze()
        radiance_nr_map = raw_diffuse_nr_map[...,0].flatten()[:,None,None].clamp(0)
        offset_nr_map       = raw_diffuse_nr_map[...,1:cfg.num_basis*3+1].reshape(-1, cfg.num_basis, 3)
        omega_nr_map        = raw_diffuse_nr_map[...,cfg.num_basis*3+1:cfg.num_basis*4+1].reshape(-1, cfg.num_basis, 1)
        omega_nr_map        = F.softplus(omega_nr_map, beta=1, threshold=20) + 0.05
        omega_nr_map        = omega_nr_map / omega_nr_map.sum(dim=1, keepdim=True)
        # diffuse_nr_map      = raw_diffuse_nr_map[...,cfg.num_basis*4+1:cfg.num_basis*4+4].reshape(-1, 1, 3)
        # diffuse_nr_map      = F.sigmoid(diffuse_nr_map)
        # mask                = raw_diffuse_nr_map[...,[cfg.num_basis*4+4]].reshape(-1, 1, 1)
        mask                = raw_diffuse_nr_map[...,cfg.num_basis*4+4].flatten()[:,None,None]
        mask                = F.sigmoid(mask)
        # vd_nr_map           = raw_viewdep_nr_map.reshape(-1, 1, 3)
        # vd_nr_map           = F.sigmoid(vd_nr_map)

        basis_color = self.net.basis_color.reshape(1, cfg.num_basis, 3)
        final_color = (radiance_nr_map*(basis_color+offset_nr_map)).clamp(0,1)
        basis_rgb = omega_nr_map*final_color
        rgb_map = basis_rgb.sum(dim=-2)

        if 'palette_basis_color' in batch:
            basis_color_val = batch['palette_basis_color']
        else:
            if len(cfg.val_cloth_color)>0:
                basis_color_val = torch.tensor([cfg.val_cloth_color,
                                            cfg.val_body_color]).cuda()
            else:

                basis_color_val = basis_color.detach().clone()[0]
                basis_color_val[cfg.val_palette_idx] = torch.tensor(cfg.val_palette_color).cuda()[cfg.val_palette_idx]


        final_color_val = (radiance_nr_map*(basis_color_val+offset_nr_map))
        basis_rgb_val = omega_nr_map*final_color_val
        rgb_map_val = basis_rgb_val.sum(dim=-2)

        rgb_map_val = rgb_map_val.clamp(0,1)
        if 'palette_basis_color' in batch:
            rgb_map = rgb_map_val


        rgb_map = torch.cat([rgb_map, mask[:,0]], dim=-1)


        ret['rgb_map'] = rgb_map[None]

        acc_map = acc_map.view(n_batch, n_pixel)
        depth_map = depth_map.view(n_batch, n_pixel)
        weights_map = weights.view(n_batch, n_pixel, -1)

        ret.update({
            'basis_rgb': basis_rgb[None],
            'rgb_map_val': rgb_map_val[None],
            'radiance_map': radiance_nr_map[None],
            'offset_map': offset_nr_map[None],
            'omega_map': omega_nr_map[None],
            'acc_map': acc_map,
        })

        if batch['mode'] == 'train':
            output = ret
        else:
            output = {
                'rgb_map': rgb_map[None]
            }
            if not cfg.nr_only:
                output['acc_map'] = acc_map

        if not rgb_map.requires_grad:
            output = {k: output[k].detach() for k in output.keys()}
        return output

    def render(self, batch):
        s0 = time.time()
        ray_o = batch['ray_o']
        ray_d = batch['ray_d']
        near = batch['near']
        far = batch['far']

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
            pixel_value = self.get_pixel_value(ray_o_chunk, ray_d_chunk,
                                               near_chunk, far_chunk,
                                               None, batch)

            if batch['mode'] != 'train':
                pixel_value = {k: pixel_value[k].detach().cpu() for k in pixel_value.keys()}
            ret_list.append(pixel_value)

        keys = ret_list[0].keys()
        ret = {k: torch.cat([r[k] for r in ret_list], dim=1) for k in keys}
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