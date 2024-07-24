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

from pytorch3d.renderer import TexturesVertex

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

    def get_upsampling_points(self, ray_o, ray_d, z_vals, batch):
        if batch['mode'] != 'train' and cfg.N_importance==0:
            n_importance = 64
        else:
            n_importance = cfg.N_importance
        # find the z value for the upsampling points
        _, batch_size, n_samples = z_vals.shape
        with torch.no_grad():
            z_vals = z_vals[0]
            pts = ray_o[:, :, None] + ray_d[:, :, None] * z_vals[..., None]
            # 1.4 GB
            sdf = self.net.sdf_decoder(pts.reshape(-1, 3), batch).reshape(batch_size, n_samples)# [65536, 1]
            # 2.6 GB
            for i in range(cfg.up_sample_steps):
                # 在重点区域采样新的点
                new_z_vals = nerf_net_utils.up_sample(ray_o,
                                            ray_d,
                                            z_vals,
                                            sdf,
                                            n_importance // cfg.up_sample_steps,
                                            64 * 2**i)
                # 结合新旧点，重新计算sdf，并更新z_vals
                z_vals, sdf = self.cat_z_vals(ray_o,
                                              ray_d,
                                              z_vals,
                                              new_z_vals,
                                              sdf,
                                              batch,
                                              last=(i + 1 == cfg.up_sample_steps))
        if cfg.importance_only:
            z_vals = z_vals[:,cfg.N_samples:]
        # 3.0 GB
        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        # todo：这里加了一个点可能是为了补齐之前因为减法运算导致的少一个点的问题
        sample_dist = 2.0 / cfg.N_samples
        dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape).to(z_vals.device)], -1)
        mid_z_vals = z_vals + dists * 0.5

        # Section midpoints
        pts = ray_o[:, :, None] + ray_d[:, :, None] * mid_z_vals[..., :, None]
        z_vals = z_vals[None]
        return pts, z_vals

    # def get_curr_batch(self, batch):
    #     curr_batch = {}
    #     for k in batch:
    #         if not k.startswith('next_'):
    #             curr_batch[k] = batch[k]
    #     for i in range(cfg.prev_frames):
    #         curr_batch[f'wverts_prev_{i+1}'] = batch[f'wverts_prev_{i+2}']
    #     return curr_batch

    def get_next_batch(self, batch):
        next_batch = {}
        for k in batch:
            if k.startswith('next_'):
                next_batch[k[5:]] = batch[k]
        for i in range(cfg.prev_frames-1):
            next_batch[f'wverts_prev_{i+2}'] = batch[f'wverts_prev_{i+1}']
        next_batch['wverts_prev_1'] = batch['wverts']
        next_batch['mode'] = batch['mode']
        return next_batch

    def get_density_color(self, wpts, viewdir, z_vals, batch, do_segment=False):
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


        # get current frame input

        if cfg.auto_regression and batch['mode'] == 'train':
            curr_batch = batch
            curr_inp,  curr_local_coordinates = self.net.prepare_input(wpts, curr_batch)
            # sdf_curr = self.net.sdf_decoder(wpts, batch, inp, local_coordinates)
            ret = self.net(curr_inp, curr_local_coordinates, viewdir, dists, curr_batch)
            if cfg.train_deform_net:
                wpts.requires_grad = True
                ret_deform = self.net.dwpts_decoder(wpts, curr_batch, curr_inp, curr_local_coordinates, get_divergence=cfg.divergence_wt > 0)
                dwpts = ret_deform['dwpts']
                if cfg.divergence_wt > 0:
                    ret['divergence'] = ret_deform['divergence'][None]

            else:
                with torch.no_grad():
                    dwpts = self.net.dwpts_decoder(wpts, curr_batch, curr_inp, curr_local_coordinates)
            ret['dwpts'] = dwpts
            wpts_next = wpts + dwpts
            # with torch.no_grad():
            next_batch = self.get_next_batch(batch)
            inp, local_coordinates = self.net.prepare_input(wpts_next, next_batch)
            sdf_next = self.net.sdf_decoder(wpts_next, next_batch, inp, local_coordinates)
            ret['sdf_next'] = sdf_next
        else:
            inp, local_coordinates = self.net.prepare_input(wpts, batch)
            ret = self.net(inp, local_coordinates, viewdir, dists, batch)
        return ret

    def get_segment_mask(self, inp, thres_upper=0.32, thres_lower=-0.23):
        z = inp[..., 1]
        mask_head = (z >= thres_upper)
        mask_upper = (z < thres_upper) & (z > thres_lower)
        mask_lower = (z <= thres_lower)
        return mask_head, mask_upper, mask_lower

    def cat_z_vals(self, rays_o, rays_d, z_vals, new_z_vals, sdf, batch, last=False):
        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape
        pts = rays_o[:, :, None] + rays_d[:, :, None] * z_vals[..., None]
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        z_vals, index = torch.sort(z_vals, dim=-1)

        if not last:
            new_sdf = self.net.sdf_decoder(pts.reshape(-1, 3), batch).reshape(batch_size, n_samples)
            sdf = torch.cat([sdf, new_sdf], dim=-1)
            xx = torch.arange(batch_size)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
            index = index.reshape(-1)
            sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)

        return z_vals, sdf

    def get_pixel_value(self, ray_o, ray_d, near, far, occ, batch):
        if cfg.surface_rendering:# and batch['mode'] == 'train':
            """
            cond_dim = 1
            cond = torch.ones([1, cond_dim], device='cuda')
            tex = self.net.mlp_tex(batch['wverts'], cond.detach())
            texture = TexturesVertex(verts_features=tex)
            body_img, body_depth_map = self.net.surface_renderer.render(batch['wverts'], batch['frame_index'], batch['cam_ind'],texture, extra_rot=batch['extra_rot'], mode=batch['mode'])
            """
            texture = None
            if cfg.rotate_camera:
                body_img, body_depth_map = self.net.surface_renderer.render_with_light(batch['wverts'],batch['b_faces'].long(), batch['frame_index'], batch['cam_ind'],texture, extra_rot=batch['extra_rot'], mode=batch['mode'])
            else:
                body_img, body_depth_map = self.net.surface_renderer.render_with_light(batch['wverts'],batch['b_faces'].long(), batch['frame_index'], batch['cam_ind'],texture, mode=batch['mode'])

            batch['body_occupancy'] = body_depth_map[batch['select_coord'][0,:, 0], batch['select_coord'][0,:, 1]] != -1
            body_occupancy = batch['body_occupancy'][batch['curr_chunk'][0]:batch['curr_chunk'][1]]
            # don't sample point after body surface
            if batch['mode'] == 'train' or cfg.render_body:
                far[:,body_occupancy] = torch.min(far[0, body_occupancy], body_depth_map[batch['select_coord'][0,:, 0], batch['select_coord'][0,:, 1]][batch['curr_chunk'][0]:batch['curr_chunk'][1]][body_occupancy])
            chunk_s, chunk_e = batch['curr_chunk']
            batch['body_rgb'] = body_img[batch['select_coord'][0,chunk_s:chunk_e, 0], batch['select_coord'][0,chunk_s:chunk_e, 1]]


            # far = torch.min(far,body_depth_map[batch['select_coord'][0, :, 0], batch['select_coord'][0, :, 1]])

            # batch['body_occupancy'] = (depth_map[batch['select_coord'][0,:, 0], batch['select_coord'][0,:, 1]] != -1)
            # res = torch.zeros(512,512)
            # res[batch['select_coord'][0,:,0]*body_occupancy.long(), batch['select_coord'][0,:,1]*body_occupancy.long()]=1


        # sampling points along camera rays
        # [1, 1024, 64, 3]
        wpts, z_vals = self.get_sampling_points(ray_o, ray_d, near, far)
        # up sample the points
        if cfg.N_importance > 0 or batch['mode']!='train':
            wpts, z_vals = self.get_upsampling_points(ray_o, ray_d, z_vals, batch)
        # viewing direction
        viewdir = ray_d

        # compute the color and density
        ret = self.get_density_color(wpts, viewdir, z_vals, batch, do_segment=self.clothes_change)

        # volume rendering for wpts
        n_batch, n_pixel, n_sample = z_vals.shape
        raw = ret['raw'].reshape(-1, n_sample, 4)
        z_vals = z_vals.view(-1, n_sample)
        ray_d = ray_d.view(-1, 3)
        if cfg.surface_rendering:# and batch['mode'] != 'train':
            # if cfg.render_body and batch['mode'] != 'train':
            #     replace_raw = torch.zeros(raw.shape[0], 1, 4).cuda()
            #     # alpha = 1
            #     replace_raw[body_occupancy, 0, 3] = 1.
            #     # rgb = surface
            #     replace_raw[body_occupancy, 0, :3] = batch['body_rgb'][batch['curr_chunk'][0]:batch['curr_chunk'][1]][body_occupancy]
            #     raw[:, -1, :] = replace_raw[:, 0, :]
            # else:
            raw[body_occupancy, -1, :3] = batch['body_rgb'][body_occupancy]


        rgb_map, disp_map, acc_map, weights, depth_map = nerf_net_utils.raw2outputs_neus(raw, z_vals, cfg.white_bkgd)

        if cfg.surface_rendering:
            rgb_map = rgb_map * acc_map[...,None] + batch['body_rgb'] * (1. - acc_map[...,None])

        rgb_map = rgb_map.view(n_batch, n_pixel, -1)
        acc_map = acc_map.view(n_batch, n_pixel)
        depth_map = depth_map.view(n_batch, n_pixel)
        weights_map = weights.view(n_batch, n_pixel, -1)

        ret.update({
            'rgb_map': rgb_map,
            'acc_map': acc_map,
            'depth_map': depth_map,
            'raw': raw.view(n_batch, -1, 4),
            'weights_map': weights_map,
        })
        if cfg.surface_rendering:
            if batch['mode'] == 'train':
                ret.update({
                    'body_img': body_img,
                    'body_depth_map': body_depth_map,
                })



        # if cfg.normal_constraint:
        # with torch.no_grad:
        # surface_mask = torch.zeros_like(raw[..., -1], device=raw.device) #[512,128]
        max_alpha_idx = raw[..., -1].max(dim=1)[1] #[512]
        # surface_mask[torch.arange(n_batch), max_alpha_idx] = 1
        gradient = ret['gradients'].reshape(-1, n_sample, 3) #[512,128,3]
        surface_normal = gradient[torch.arange(n_batch*n_pixel), max_alpha_idx] #[512,3]
        ret['surface_normal'] = surface_normal.reshape(n_batch, n_pixel, 3)

        if cfg.gt_constraint:
            pred_depth_map = z_vals[torch.arange(n_batch*n_pixel), max_alpha_idx]
            ret['pred_depth_map'] = pred_depth_map.reshape(n_batch, n_pixel)

        #     # compute the surface normal
        #     with torch.no_grad():
        #         points, network_object_mask, dists = self.ray_tracer(sdf=lambda x: self.net.sdf_decoder(x, batch).squeeze(),
        #                                                              cam_loc=ray_o[:,0],
        #                                                              object_mask=batch['occupancy'][0]==1,
        #                                                              ray_directions=ray_d)
        if 'sdf' in ret:
            if cfg.msk_loss_type == 'crit':
                # get pixels that outside the mask or no ray-geometry intersection
                sdf = ret['sdf'].view(n_batch, n_pixel, n_sample)
                min_sdf = sdf.min(dim=2)[0] # 每条光线上最小的sdf值
                free_sdf = min_sdf[occ == 0] # 人体以外的光线（GT）
                free_label = torch.zeros_like(free_sdf) # 人体以外的光线都应该为0

                with torch.no_grad():
                    intersection_mask, _ = nerf_net_utils.get_intersection_mask(sdf, z_vals) # 与人体相交的光线(pred)
                ind = (intersection_mask == False) * (occ == 1) # 与人体不相交，但应该相交的光线
                sdf = min_sdf[ind] #　与人体不相交，但应该相交的光线的sdf值
                label = torch.ones_like(sdf) # 与人体不相交，但应该相交的光线的label为1

                sdf = torch.cat([sdf, free_sdf])
                label = torch.cat([label, free_label])
                ret.update({
                    'msk_sdf': sdf.view(n_batch, -1),
                    'msk_label': label.view(n_batch, -1)
                })
            elif cfg.msk_loss_type == 'l1':

                sdf = ret['sdf'].view(n_batch, n_pixel, n_sample)
                ind = (sdf <=0) * (occ != 1)[..., None]
                outside_sdf = sdf[ind]   # should be positive
                # inside_sdf = sdf.min(dim=2)[0][occ == 1 & sdf >0]  # should be negative or zero
                ret.update({
                    'outside_sdf': outside_sdf.view(n_batch, -1),
                    # 'inside_sdf': inside_sdf.view(n_batch, -1)
                })
            elif cfg.msk_loss_type == 'bce':
                pass

        if 'pred_vel' in ret:
            pred_vel = ret['pred_vel'].view(n_batch, n_pixel, -1)
            sdf = ret['sdf'].view(n_batch, n_pixel, n_sample)
            sign = torch.sign(sdf[..., :-1] * sdf[..., 1:])



        # if cfg.normal_constraint:
        with torch.no_grad():
            sdf = ret['sdf'].reshape(-1, n_sample)
            inside_points = sdf < 5.0e-5

            first_zero_sdf_idx = torch.argmax(inside_points.long(), dim=1)
            wpts = ret['wpts'].reshape(-1, n_sample, 3)
            surface_points = wpts[torch.arange(n_batch*n_pixel), first_zero_sdf_idx] #[512,3]


            no_intersect_mask = torch.sum(inside_points.long(),dim=1)!=0 # [512]
            max_alpha_idx = raw[..., -1].max(dim=1)[1] #[512]
            # surface_mask[torch.arange(n_batch), max_alpha_idx] = 1
        gradient = ret['gradients'].reshape(-1, n_sample, 3) #[512,128,3]
        surface_normal = gradient[torch.arange(n_batch*n_pixel), first_zero_sdf_idx] #[512,3]
        ret['surface_normal'] = surface_normal.reshape(n_batch, n_pixel, 3)
        ret['no_intersect_mask'] = no_intersect_mask.reshape(n_batch, n_pixel)
        ret['surface_points'] = surface_points.reshape(n_batch, n_pixel, 3)
        if batch['mode'] == 'train':
            output = ret
        else:
            output = {
                'rgb_map': ret["rgb_map"],
                'sdf': ret['sdf'],
            }

            # if cfg.normal_constraint:
            output['surface_normal'] = ret['surface_normal']
            output['no_intersect_mask'] = ret['no_intersect_mask']
            output['surface_points'] = ret['surface_points']
            output['weights_map'] = ret['weights_map']
            output['acc_map'] = ret['acc_map']


        if not rgb_map.requires_grad:
            output = {k: output[k].detach() for k in output.keys()}

        return output

    def render(self, batch):
        ray_o = batch['ray_o']
        ray_d = batch['ray_d']
        near = batch['near']
        far = batch['far']
        occ = batch['occupancy']
        
        # volume rendering for each pixel
        n_batch, n_pixel = ray_o.shape[:2]
        chunk = cfg.chunk
        ret_list = []
        for i in range(0, n_pixel, chunk):
            ray_o_chunk = ray_o[:, i:i + chunk]
            ray_d_chunk = ray_d[:, i:i + chunk]
            near_chunk = near[:, i:i + chunk]
            far_chunk = far[:, i:i + chunk]
            occ_chunk = occ[:, i:i + chunk]
            batch['curr_chunk'] = [i, i + chunk]
            pixel_value = self.get_pixel_value(ray_o_chunk, ray_d_chunk,
                                               near_chunk, far_chunk,
                                               occ_chunk, batch)
            if batch['mode'] != 'train':
                pixel_value = {k: pixel_value[k].detach().cpu() for k in pixel_value.keys()}
            ret_list.append(pixel_value)

        keys = ret_list[0].keys()
        ret = {k: torch.cat([r[k] for r in ret_list], dim=1) for k in keys}


        # import matplotlib.pyplot as plt
        #
        # plot_sdf = ret['sdf'].reshape(-1,128).detach().cpu()
        # sdf_max = ret['sdf'].max().detach().cpu()
        # plot_coord = batch['select_coord'][0].detach().cpu()
        # import cv2
        # for ij in range(128):
        #     img = torch.zeros(512,512)
        #     img[plot_coord[:,0],plot_coord[:,1]] = plot_sdf[:,ij]/sdf_max
        #     cv2.imwrite(f'{ij}.jpg',img.numpy()*255)
            # plt.axis('off')
            # plt.imsave(img,f'{ij}.jpg')
            # plt.close()
            #

            # plt.imshow(img)
            # plt.show()


        return ret

    def get_mesh(self, batch):
        for k in batch:
            if torch.is_tensor(batch[k]):#k != 'meta' and k != 'iter_step':
                batch[k] = batch[k].cuda()
        with torch.no_grad():
            output = self.mesh_renderer.render(batch)
            self.mesh_visualizer.visualize(output, batch)

    def get_nerf_mesh(self, batch):
        for k in batch:
            if torch.is_tensor(batch[k]):#k != 'meta' and k != 'iter_step':
                batch[k] = batch[k].cuda()
        with torch.no_grad():
            output = self.mesh_renderer.render(batch, get_alpha=True)
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