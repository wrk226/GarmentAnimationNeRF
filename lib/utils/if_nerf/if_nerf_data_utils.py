import matplotlib.pyplot as plt
import numpy as np
import torch
from math import sqrt, exp

from lib.utils import base_utils
import cv2
from lib.config import cfg
import trimesh
import os

class RaySampler(object):
    def __init__(self, N_samples):
        super(RaySampler, self).__init__()
        self.N_samples = N_samples
        self.scale = torch.ones(1,).float()
        self.return_indices = True

        # Ray helpers
    def get_rays(self, H, W, focal, c2w):
        i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
        i = i.t()
        j = j.t()
        dirs = torch.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -torch.ones_like(i)], -1)
        # Rotate ray directions from camera frame to the world frame
        rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
        # Translate camera frame's origin to the world frame. It is the origin of all rays.
        rays_o = c2w[:3,-1].expand(rays_d.shape)
        return rays_o, rays_d

    def __call__(self, H, W, focal, pose):
        rays_o, rays_d = self.get_rays(H, W, focal, pose)

        select_inds = self.sample_rays(H, W)

        if self.return_indices:
            rays_o = rays_o.view(-1, 3)[select_inds]
            rays_d = rays_d.view(-1, 3)[select_inds]

            h = (select_inds // W) / float(H) - 0.5
            w = (select_inds %  W) / float(W) - 0.5

            hw = torch.stack([h,w]).t()

        else:
            rays_o = torch.nn.functional.grid_sample(rays_o.permute(2,0,1).unsqueeze(0),
                                 select_inds.unsqueeze(0), mode='bilinear', align_corners=True)[0]
            rays_d = torch.nn.functional.grid_sample(rays_d.permute(2,0,1).unsqueeze(0),
                                 select_inds.unsqueeze(0), mode='bilinear', align_corners=True)[0]
            rays_o = rays_o.permute(1,2,0).view(-1, 3)
            rays_d = rays_d.permute(1,2,0).view(-1, 3)

            hw = select_inds
            select_inds = None

        return torch.stack([rays_o, rays_d]), select_inds, hw

    def sample_rays(self, H, W):
        raise NotImplementedError


class FlexGridRaySampler(RaySampler):
    def __init__(self, N_samples, random_shift=True, random_scale=True, min_scale=0.25, max_scale=1., scale_anneal=-1.,
                 **kwargs):
        self.N_samples_sqrt = int(sqrt(N_samples))
        super(FlexGridRaySampler, self).__init__(self.N_samples_sqrt**2)

        self.random_shift = random_shift
        self.random_scale = random_scale

        self.min_scale = min_scale
        self.max_scale = max_scale

        # nn.functional.grid_sample grid value range in [-1,1]
        self.w, self.h = torch.meshgrid([torch.linspace(-1,1,self.N_samples_sqrt),
                                         torch.linspace(-1,1,self.N_samples_sqrt)])
        self.h = self.h.unsqueeze(2)
        self.w = self.w.unsqueeze(2)

        # directly return grid for grid_sample
        self.return_indices = False

        self.iterations = 0
        self.scale_anneal = scale_anneal

    def sample_rays(self, H, W):

        if self.scale_anneal>0:
            k_iter = self.iterations // 1000 * 3
            min_scale = max(self.min_scale, self.max_scale * exp(-k_iter*self.scale_anneal))
            min_scale = min(0.9, min_scale)
        else:
            min_scale = self.min_scale

        # scale = H/self.h
        # random_scale:
        scale = torch.Tensor(1).uniform_(min_scale, self.max_scale)
        h = self.h * scale
        w = self.w * scale

        # random_shift:
        max_offset = 1-scale.item()
        h_offset = torch.Tensor(1).uniform_(0, max_offset) * (torch.randint(2,(1,)).float()-0.5)*2
        w_offset = torch.Tensor(1).uniform_(0, max_offset) * (torch.randint(2,(1,)).float()-0.5)*2

        h += h_offset
        w += w_offset

        self.scale = scale

        return ((torch.cat([h, w], dim=2)+1)/2*(H-1)).long()

def get_rays_within_bounds_test(H, W, K, R, T, bounds):
    ray_o, ray_d = get_rays(H, W, K, R, T)

    ray_o = ray_o.reshape(-1, 3).astype(np.float32)
    ray_d = ray_d.reshape(-1, 3).astype(np.float32)
    near, far, mask_at_box = get_near_far(bounds, ray_o, ray_d)
    near = near.astype(np.float32)
    far = far.astype(np.float32)
    ray_o = ray_o.reshape(H, W, 3)
    ray_d = ray_d.reshape(H, W, 3)

    mask_at_box = mask_at_box.reshape(H, W)

    return ray_o, ray_d, near, far, mask_at_box


def get_rays(H, W, K, R, T):
    # calculate the camera origin
    rays_o = -np.dot(R.T, T).ravel()
    # calculate the world coodinates of pixels
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32),
                       indexing='xy')
    xy1 = np.stack([i, j, np.ones_like(i)], axis=2)
    pixel_camera = np.dot(xy1, np.linalg.inv(K).T)
    pixel_world = np.dot(pixel_camera - T.ravel(), R)
    # calculate the ray direction
    rays_d = pixel_world - rays_o[None, None]
    rays_d = rays_d / np.linalg.norm(rays_d, axis=2, keepdims=True)
    rays_o = np.broadcast_to(rays_o, rays_d.shape)
    # todo：是为了避免后面的除零错误，需要修改
    return rays_o+1e-5, rays_d+1e-5
    # return rays_o, rays_d


def get_bound_corners(bounds):
    min_x, min_y, min_z = bounds[0]
    max_x, max_y, max_z = bounds[1]
    corners_3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corners_3d


def get_bound_2d_mask(bounds, K, pose, H, W):
    corners_3d = get_bound_corners(bounds)
    corners_2d = base_utils.project(corners_3d, K, pose)
    corners_2d = np.round(corners_2d).astype(int)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 3, 2, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[4, 5, 7, 6, 5]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 5, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[2, 3, 7, 6, 2]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 2, 6, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[1, 3, 7, 5, 1]]], 1)
    return mask


# def get_near_far(bounds, ray_o, ray_d):
#     """calculate intersections with 3d bounding box"""
#     norm_d = np.linalg.norm(ray_d, axis=-1, keepdims=True)
#     viewdir = ray_d / norm_d
#     viewdir[(viewdir < 1e-5) & (viewdir > -1e-10)] = 1e-5
#     viewdir[(viewdir > -1e-5) & (viewdir < 1e-10)] = -1e-5
#     tmin = (bounds[:1] - ray_o[:1]) / viewdir
#     tmax = (bounds[1:2] - ray_o[:1]) / viewdir
#     t1 = np.minimum(tmin, tmax)
#     t2 = np.maximum(tmin, tmax)
#     near = np.max(t1, axis=-1)
#     far = np.min(t2, axis=-1)
#     mask_at_box = near < far
#     near = near[mask_at_box] / norm_d[mask_at_box, 0]
#     far = far[mask_at_box] / norm_d[mask_at_box, 0]
#     return near, far, mask_at_box

def get_near_far_general(bounds, point):
    min_x, min_y, min_z, max_x, max_y, max_z = bounds.ravel()
    # Define the eight vertices of the bounding box
    vertices = np.array([
        [min_x, min_y, min_z],

        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [max_x, min_y, min_z],

        [min_x, max_y, max_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],

        [max_x, max_y, max_z]
    ])


    # Compute the distances from the point to each vertex
    distances = np.linalg.norm(vertices - point, axis=1)

    # Nearest and Farthest distances
    nearest_distance = np.min(distances)
    farthest_distance = np.max(distances)

    # For the nearest distance, we also check if the point is outside the bounding box
    # and apply the axis-aligned bounding box method.
    min_x, min_y, min_z, max_x, max_y, max_z = bounds.ravel()

    # Shift and scale the space so that the bounding box becomes (-1, -1, -1) to (1, 1, 1)
    x = 2 * (point[0] - min_x) / (max_x - min_x) - 1
    y = 2 * (point[1] - min_y) / (max_y - min_y) - 1
    z = 2 * (point[2] - min_z) / (max_z - min_z) - 1

    # Compute the nearest distance using the compact formula
    distance_outside_box = np.sqrt(
        max(0, abs(x) - 1) ** 2 +
        max(0, abs(y) - 1) ** 2 +
        max(0, abs(z) - 1) ** 2
    )

    # Scale back the distance to original space
    scale_factor = (max_x - min_x) / 2
    distance_outside_box *= scale_factor

    nearest_distance = min(nearest_distance, distance_outside_box)

    return nearest_distance, farthest_distance

def get_near_far(bounds, ray_o, ray_d, use_full_patch=False):
    """calculate intersections with 3d bounding box"""
    bounds = bounds + np.array([-0.01, 0.01])[:, None]
    nominator = bounds[None] - ray_o[:, None]
    # calculate the step of intersections at six planes of the 3d bounding box
    d_intersect = (nominator / ray_d[:, None]).reshape(-1, 6)
    # calculate the six interections
    p_intersect = d_intersect[..., None] * ray_d[:, None] + ray_o[:, None]
    # calculate the intersections located at the 3d bounding box
    min_x, min_y, min_z, max_x, max_y, max_z = bounds.ravel()
    eps = 1e-6
    p_mask_at_box = (p_intersect[..., 0] >= (min_x - eps)) * \
                    (p_intersect[..., 0] <= (max_x + eps)) * \
                    (p_intersect[..., 1] >= (min_y - eps)) * \
                    (p_intersect[..., 1] <= (max_y + eps)) * \
                    (p_intersect[..., 2] >= (min_z - eps)) * \
                    (p_intersect[..., 2] <= (max_z + eps))
    # obtain the intersections of rays which intersect exactly twice
    mask_at_box = p_mask_at_box.sum(-1) == 2
    p_intervals = p_intersect[mask_at_box][p_mask_at_box[mask_at_box]].reshape(
        -1, 2, 3)

    # calculate the step of intersections

    ray_o_intersect = ray_o[mask_at_box]
    ray_d_intersect = ray_d[mask_at_box]
    norm_ray = np.linalg.norm(ray_d_intersect, axis=1)
    d0 = np.linalg.norm(p_intervals[:, 0] - ray_o_intersect, axis=1) / norm_ray
    d1 = np.linalg.norm(p_intervals[:, 1] - ray_o_intersect, axis=1) / norm_ray
    if use_full_patch:
        distance = np.concatenate([d0, d1]).reshape(-1)
        dist_min, dist_max = np.min(distance), np.max(distance)
        near = np.ones_like(ray_o[:, 0]) * dist_min
        far = np.ones_like(ray_o[:, 0]) * dist_max
    else:
        near = np.minimum(d0, d1)
        far = np.maximum(d0, d1)
    #      512,   512,  512
    return near, far, mask_at_box

def sample_ray_h36m_full(img, msk, K, R, T, bounds, nrays, split):
    patch_info = {}
    H, W = img.shape[:2]
    res = np.sqrt(nrays).astype(np.int32)
    ray_o, ray_d = get_rays(H, W, K, R, T)
    mask_at_box = np.ones_like(img[..., 0])
    ray_o = ray_o[::H//res, ::W//res]
    ray_d = ray_d[::H//res, ::W//res]
    img = img[::H//res, ::W//res]
    mask_at_box = mask_at_box[::H//res, ::W//res]


    rgb = img.reshape(-1, 3).astype(np.float32)
    ray_o = ray_o.reshape(-1, 3).astype(np.float32)
    ray_d = ray_d.reshape(-1, 3).astype(np.float32)
    # near, far, mask_at_box = get_near_far(bounds, ray_o, ray_d)
    near, far = get_near_far_general(bounds, ray_o[0])
    near, far = np.ones([ray_o.shape[0]])*near, np.ones([ray_o.shape[0]])*far
    near = near.astype(np.float32)
    far = far.astype(np.float32)
    rgb = rgb
    ray_o = ray_o
    ray_d = ray_d

    coord = np.argwhere(mask_at_box.reshape(res, res) > -1)

    return rgb, ray_o, ray_d, near, far, coord, mask_at_box, patch_info

def sample_ray_h36m_grid(img, msk, K, R, T, bounds, nrays, split, grid_sampler):
    patch_info = {}
    H, W = img.shape[:2]
    res = np.sqrt(nrays).astype(np.int32)
    ray_o, ray_d = get_rays(H, W, K, R, T)
    grid_sampler.iterations += 1
    select_pixel_inds = grid_sampler.sample_rays(H, W).reshape(-1,2).long()[:,[1,0]]
    ray_o = ray_o[select_pixel_inds[:,0],select_pixel_inds[:,1]]
    ray_d = ray_d[select_pixel_inds[:,0],select_pixel_inds[:,1]]
    img = img[select_pixel_inds[:,0],select_pixel_inds[:,1]]

    rgb = img.reshape(-1, 3).astype(np.float32)
    ray_o = ray_o.reshape(-1, 3).astype(np.float32)
    ray_d = ray_d.reshape(-1, 3).astype(np.float32)
    # near, far, mask_at_box = get_near_far(bounds, ray_o, ray_d)
    near, far = get_near_far_general(bounds, ray_o[0])
    near, far = np.ones([ray_o.shape[0]])*near, np.ones([ray_o.shape[0]])*far
    near = near.astype(np.float32)
    far = far.astype(np.float32)
    rgb = rgb
    ray_o = ray_o
    ray_d = ray_d
    mask_at_box = np.ones([res*res])#.astype(np.int64)
    coord = select_pixel_inds

    return rgb, ray_o, ray_d, near, far, coord, mask_at_box, patch_info

def get_mask_at_box(img, msk, K, R, T, bounds):
    H, W = img.shape[:2]
    pose = np.concatenate([R, T], axis=1)
    bound_mask = get_bound_2d_mask(bounds, K, pose, H, W)
    return bound_mask

def sample_ray_h36m_bbox(img, msk, K, R, T, bounds, nrays, split, use_full_patch=False):
    patch_info = {}
    H, W = img.shape[:2]
    # ray_o, ray_d = get_rays(H, W, K, R, T)

    pose = np.concatenate([R, T], axis=1)
    bound_mask = get_bound_2d_mask(bounds, K, pose, H, W)
    # if cfg.mask_bkgd:
    #     img[bound_mask != 1] = 0
    # if cfg.surface_rendering:
    #     msk = msk == 255
    msk = bound_mask


    patch_info = {}
    H, W = img.shape[:2]
    res = np.sqrt(nrays).astype(np.int32)
    assert H//res==W//res
    scale = H//res
    # mask_at_box = msk[::scale, ::scale]

    # coords_1d = coords_2d[:, 0] * H + coords_2d[:, 1]

    K = K/scale
    K[2, 2] = 1
    ray_o, ray_d = get_rays(res, res, K, R, T)
    img = img[::scale, ::scale]

    rgb = img.reshape(-1, 3).astype(np.float32)
    ray_o = ray_o.reshape(-1, 3).astype(np.float32)
    ray_d = ray_d.reshape(-1, 3).astype(np.float32)
    near, far, mask_at_box = get_near_far(bounds, ray_o, ray_d)

    coords_2d = np.argwhere(mask_at_box.reshape(res,res) > 0)
    ray_o = ray_o[mask_at_box]
    ray_d = ray_d[mask_at_box]
    near = near.astype(np.float32)
    far = far.astype(np.float32)
    # print(near_.shape)

    # min_dhw, max_dhw = bounds[:,[2,1,0]]
    # voxel_size = np.array(cfg.voxel_size)
    # coord = np.round((dhw - min_dhw) / voxel_size).astype(np.int32)
    # # construct the output shape
    # out_sh = np.ceil((max_dhw - min_dhw) / voxel_size).astype(np.int32)
    # x = 32
    # # 保证输出的形状是32的整数倍
    # out_sh = (out_sh | (x - 1)) + 1





    return rgb, ray_o, ray_d, near, far, coords_2d, mask_at_box, patch_info

def sample_ray_h36m(img, msk, K, R, T, bounds, nrays, split, use_full_patch=False):
    patch_info = {}
    H, W = img.shape[:2]
    ray_o, ray_d = get_rays(H, W, K, R, T)

    pose = np.concatenate([R, T], axis=1)
    bound_mask = get_bound_2d_mask(bounds, K, pose, H, W)
    if cfg.mask_bkgd:
        img[bound_mask != 1] = 0
    # if cfg.surface_rendering:
    #     msk = msk == 255
    msk = msk * bound_mask
    cloth_msk = msk == 6 #255

    if split == 'train':
        nsampled_rays = 0
        if cfg.patch_sample:
            cloth_sample_ratio = 0.
            body_sample_ratio = 0.
        else:
            cloth_sample_ratio = cfg.cloth_sample_ratio
            body_sample_ratio = cfg.body_sample_ratio
        ray_o_list = []
        ray_d_list = []
        rgb_list = []
        near_list = []
        far_list = []
        coord_list = []
        mask_at_box_list = []
        coord_init = None

        if cfg.patch_sample:
            half_size = cfg.sample_patch_size//2
            while True:
                # 不断采样小图像块，直到采样到的图像块中有足够的点(1/3以上)就使用这个图像块
                tmp_mask = np.zeros([H, W])
                center = np.random.randint(low=half_size*2, high=H-half_size*2, size=(2,))
                tmp_mask[center[0]-half_size:center[0]+half_size, center[1]-half_size:center[1]+half_size] = 1.
                inds = np.nonzero(tmp_mask)
                # gt_mask = np.clip(cloth_msk, a_min=None, a_max=1)
                gt_mask = np.clip(bound_mask, a_min=None, a_max=1)

                patch_mask = gt_mask[inds[0], inds[1]]
                if patch_mask.sum() > cfg.sample_patch_size**2/2.:#32*32:
                    # gt_mask_tmp = np.copy(cloth_msk)
                    # tmp_mask[center[0]-half_size:center[0]+half_size, center[1]-half_size:center[1]+half_size] \
                    #     = gt_mask_tmp[center[0]-half_size:center[0]+half_size, center[1]-half_size:center[1]+half_size]
                    # inds = np.nonzero(tmp_mask)
                    patch_info['patch_range'] = np.array([center[0]-half_size,center[0]+half_size, center[1]-half_size,center[1]+half_size])
                    patch_info['patch_inds'] = np.stack(inds, axis=1)
                    break
                    # bound_mask_temp = bound_mask>=1.
                    # patch_bounding_mask = bound_mask_temp[inds[0], inds[1]]
                    # if patch_bounding_mask.sum() == cfg.sample_patch_size**2:
                    #     break
            # 将当前图像块的内容全都替换到采样点中，保证当前图像块的内容都会被采样到
            # coord[:cfg.sample_patch_size**2] = np.stack(inds, axis=1)
            coord_init = np.stack(inds, axis=1)
            select_mask = torch.zeros_like(torch.from_numpy(msk))
            select_mask[inds[0], inds[1]] = 1.
        # plt.imshow(select_mask)
        # plt.show()
        while nsampled_rays < nrays:
            n_body = int((nrays - nsampled_rays) * body_sample_ratio)
            n_cloth = int((nrays - nsampled_rays) * cloth_sample_ratio)
            n_rand = (nrays - nsampled_rays) - n_body - n_cloth
            # sample rays on body
            coord_body = np.argwhere((msk > 0) & (msk < 6))
            coord_body = coord_body[np.random.randint(0, len(coord_body), n_body)]
            # sample rays on cloth
            coord_cloth = np.argwhere(msk == 6)
            coord_cloth = coord_cloth[np.random.randint(0, len(coord_cloth), n_cloth)]

            # sample rays in the bound mask
            if use_full_patch:
                coord = np.argwhere(select_mask > 0)
            else:
                coord = np.argwhere(bound_mask == 1)
            if coord_init is not None:
                n_rand = n_rand - len(coord_init)
                # print('n_rand1: ', n_rand)
                coord = coord[np.random.randint(0, len(coord), n_rand)]
                if len(coord) > 0:
                    coord = np.concatenate([coord_init, coord], axis=0)
                else:
                    coord = coord_init
                coord_init = None
            else:
                # print('n_rand2: ', n_rand)
                coord = coord[np.random.randint(0, len(coord), n_rand)]
            if len(coord_body)>0:
                coord = np.concatenate([coord_body, coord], axis=0)
            if len(coord_cloth) >0:
                coord = np.concatenate([coord_cloth, coord], axis=0)
            # print(coord.shape)
            ray_o_ = ray_o[coord[:, 0], coord[:, 1]]
            ray_d_ = ray_d[coord[:, 0], coord[:, 1]]
            rgb_ = img[coord[:, 0], coord[:, 1]]

            near_, far_, mask_at_box = get_near_far(bounds, ray_o_, ray_d_, use_full_patch=use_full_patch)
            ################################
            # select_coord = coord[:cfg.sample_patch_size**2]
            # # import numpy as np
            # show_gt = np.zeros((512,512,3))
            # show_gt[select_coord[..., 0], select_coord[..., 1]] = 1
            # show_pred = np.zeros((512,512,3))
            # show_pred[select_coord[..., 0], select_coord[..., 1]] = 1
            # import matplotlib.pyplot as plt
            # fig = plt.figure()
            # ax1 = fig.add_subplot(1, 2, 1)
            # ax1.imshow(show_gt)
            # ax2 = fig.add_subplot(1, 2, 2)
            # ax2.imshow(show_pred)
            # plt.show()
            ################################
            if use_full_patch:
                ray_o_list.append(ray_o_)
                ray_d_list.append(ray_d_)
                rgb_list.append(rgb_)
                near_list.append(near_)
                far_list.append(far_)
                coord_list.append(coord)
                mask_at_box_list.append(mask_at_box)
                nsampled_rays += len(near_)
            else:
                ray_o_list.append(ray_o_[mask_at_box])
                ray_d_list.append(ray_d_[mask_at_box])
                rgb_list.append(rgb_[mask_at_box])
                near_list.append(near_)
                far_list.append(far_)
                coord_list.append(coord[mask_at_box])
                mask_at_box_list.append(mask_at_box[mask_at_box])
                nsampled_rays += len(near_)

        ray_o = np.concatenate(ray_o_list).astype(np.float32)
        ray_d = np.concatenate(ray_d_list).astype(np.float32)
        rgb = np.concatenate(rgb_list).astype(np.float32)
        near = np.concatenate(near_list).astype(np.float32)
        far = np.concatenate(far_list).astype(np.float32)
        coord = np.concatenate(coord_list)
        mask_at_box = np.concatenate(mask_at_box_list)


    else:
        rgb = img.reshape(-1, 3).astype(np.float32)
        ray_o = ray_o.reshape(-1, 3).astype(np.float32)
        ray_d = ray_d.reshape(-1, 3).astype(np.float32)
        near, far, mask_at_box = get_near_far(bounds, ray_o, ray_d, use_full_patch=cfg.patch_sample)

        near = near.astype(np.float32)
        far = far.astype(np.float32)
        if cfg.patch_sample:
            assert cfg.H==cfg.W
            assert cfg.sample_patch_size**2 == cfg.N_rand
            assert cfg.chunk == cfg.N_rand
            coord = get_submatrices_indices((int(cfg.H*cfg.ratio), int(cfg.W*cfg.ratio)), (cfg.sample_patch_size,cfg.sample_patch_size))
            img_size = int(cfg.H*cfg.ratio)
            rgb = rgb[coord[:, 0]*img_size+ coord[:, 1]]
            ray_o = ray_o[coord[:, 0]*img_size+  coord[:, 1]]
            ray_d = ray_d[coord[:, 0]*img_size+  coord[:, 1]]
            near = near[coord[:, 0]*img_size+  coord[:, 1]]
            far = far[coord[:, 0]*img_size+  coord[:, 1]]
            mask_at_box = mask_at_box[coord[:, 0]*img_size+  coord[:, 1]]
            # coord = np.argwhere(mask_at_box.reshape(H, W) > -1)
        else:
            rgb = rgb[mask_at_box]
            ray_o = ray_o[mask_at_box]
            ray_d = ray_d[mask_at_box]
            coord = np.argwhere(mask_at_box.reshape(H, W) == 1)




    return rgb, ray_o, ray_d, near, far, coord, mask_at_box, patch_info

def get_submatrices_indices(matrix_shape, submatrix_shape):
    """
    分割矩阵并获得子矩阵的索引值。

    :param matrix_shape: 原始矩阵的形状，例如(512, 512)
    :param submatrix_shape: 子矩阵的形状，例如(128, 128)
    :return: 子矩阵的索引值列表
    """
    # 检查矩阵大小和子矩阵大小是否合适
    if matrix_shape[0] % submatrix_shape[0] != 0 or matrix_shape[1] % submatrix_shape[1] != 0:
        raise ValueError("原始矩阵大小不能被子矩阵大小整除。")

    num_rows = matrix_shape[0] // submatrix_shape[0]
    num_cols = matrix_shape[1] // submatrix_shape[1]

    submatrices_indices = []

    for i in range(num_rows):
        for j in range(num_cols):
            row_start, row_end = i * submatrix_shape[0], (i + 1) * submatrix_shape[0]
            col_start, col_end = j * submatrix_shape[1], (j + 1) * submatrix_shape[1]

            submatrix_indices = np.array(np.meshgrid(
                np.arange(row_start, row_end),
                np.arange(col_start, col_end)
            )).T.reshape(-1, 2)

            submatrices_indices.append(submatrix_indices)

    return np.concatenate(submatrices_indices, axis=0)

def get_rays_within_bounds(H, W, K, R, T, bounds):
    ray_o, ray_d = get_rays(H, W, K, R, T)

    ray_o = ray_o.reshape(-1, 3).astype(np.float32)
    ray_d = ray_d.reshape(-1, 3).astype(np.float32)
    near, far, mask_at_box = get_near_far(bounds, ray_o, ray_d)
    near = near.astype(np.float32)
    far = far.astype(np.float32)
    ray_o = ray_o[mask_at_box]
    ray_d = ray_d[mask_at_box]

    mask_at_box = mask_at_box.reshape(H, W)

    return ray_o, ray_d, near, far, mask_at_box


def get_acc(coord, msk):
    acc = msk[coord[:, 0], coord[:, 1]]
    acc = (acc != 0).astype(np.uint8)
    return acc


def unproject(depth, K, R, T):
    H, W = depth.shape
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32),
                       indexing='xy')
    xy1 = np.stack([i, j, np.ones_like(i)], axis=2)
    xyz = xy1 * depth[..., None]
    pts3d = np.dot(xyz, np.linalg.inv(K).T)
    pts3d = np.dot(pts3d - T.ravel(), R)
    return pts3d


def sample_world_points(ray_o, ray_d, near, far, split):
    # calculate the steps for each ray
    t_vals = np.linspace(0., 1., num=cfg.N_samples)
    z_vals = near[..., None] * (1. - t_vals) + far[..., None] * t_vals

    if cfg.perturb > 0. and split == 'train':
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = np.concatenate([mids, z_vals[..., -1:]], -1)
        lower = np.concatenate([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = np.random.rand(*z_vals.shape)
        z_vals = lower + (upper - lower) * t_rand

    pts = ray_o[:, None] + ray_d[:, None] * z_vals[..., None]
    pts = pts.astype(np.float32)
    z_vals = z_vals.astype(np.float32)

    return pts, z_vals


def barycentric_interpolation(val, coords):
    """
    :param val: verts x 3 x d input matrix
    :param coords: verts x 3 barycentric weights array
    :return: verts x d weighted matrix
    """
    t = val * coords[..., np.newaxis]
    ret = t.sum(axis=1)
    return ret


def batch_rodrigues(poses):
    """ poses: N x 3
    """
    batch_size = poses.shape[0]
    angle = np.linalg.norm(poses + 1e-8, axis=1, keepdims=True)
    rot_dir = poses / angle

    cos = np.cos(angle)[:, None]
    sin = np.sin(angle)[:, None]

    rx, ry, rz = np.split(rot_dir, 3, axis=1)
    zeros = np.zeros([batch_size, 1])
    K = np.concatenate([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros],
                       axis=1)
    K = K.reshape([batch_size, 3, 3])

    ident = np.eye(3)[None]
    rot_mat = ident + sin * K + (1 - cos) * np.matmul(K, K)

    return rot_mat


def get_rigid_transformation(poses, joints, parents, return_joints=False):
    """
    poses: 24 x 3
    joints: 24 x 3
    parents: 24
    """
    rot_mats = batch_rodrigues(poses)

    # obtain the relative joints
    rel_joints = joints.copy()
    rel_joints[1:] -= joints[parents[1:]]

    # create the transformation matrix
    transforms_mat = np.concatenate([rot_mats, rel_joints[..., None]], axis=2)
    padding = np.zeros([24, 1, 4])
    padding[..., 3] = 1
    transforms_mat = np.concatenate([transforms_mat, padding], axis=1)

    # rotate each part
    transform_chain = [transforms_mat[0]]
    for i in range(1, parents.shape[0]):
        curr_res = np.dot(transform_chain[parents[i]], transforms_mat[i])
        transform_chain.append(curr_res)
    transforms = np.stack(transform_chain, axis=0)

    posed_joints = transforms[:, :3, 3].copy()

    # obtain the rigid transformation
    padding = np.zeros([24, 1])
    joints_homogen = np.concatenate([joints, padding], axis=1)
    rel_joints = np.sum(transforms * joints_homogen[:, None], axis=2)
    transforms[..., 3] = transforms[..., 3] - rel_joints
    transforms = transforms.astype(np.float32)

    if return_joints:
        return transforms, posed_joints
    else:
        return transforms


def padding_bbox(bbox, img):
    padding = 10
    bbox[0] = bbox[0] - 10
    bbox[1] = bbox[1] + 10

    height = bbox[1, 1] - bbox[0, 1]
    width = bbox[1, 0] - bbox[0, 0]
    # a magic number of pytorch3d
    ratio = 1.5

    if height / width > ratio:
        min_size = int(height / ratio)
        if width < min_size:
            padding = (min_size - width) // 2
            bbox[0, 0] = bbox[0, 0] - padding
            bbox[1, 0] = bbox[1, 0] + padding

    if width / height > ratio:
        min_size = int(width / ratio)
        if height < min_size:
            padding = (min_size - height) // 2
            bbox[0, 1] = bbox[0, 1] - padding
            bbox[1, 1] = bbox[1, 1] + padding

    h, w = img.shape[:2]
    bbox[:, 0] = np.clip(bbox[:, 0], a_min=0, a_max=w - 1)
    bbox[:, 1] = np.clip(bbox[:, 1], a_min=0, a_max=h - 1)

    return bbox


def crop_image_msk(img, msk, K, ref_msk):
    x, y, w, h = cv2.boundingRect(ref_msk)
    bbox = np.array([[x, y], [x + w, y + h]])
    bbox = padding_bbox(bbox, img)

    crop = img[bbox[0, 1]:bbox[1, 1], bbox[0, 0]:bbox[1, 0]]
    crop_msk = msk[bbox[0, 1]:bbox[1, 1], bbox[0, 0]:bbox[1, 0]]

    # calculate the shape
    shape = crop.shape
    x = 8
    height = (crop.shape[0] | (x - 1)) + 1
    width = (crop.shape[1] | (x - 1)) + 1

    # align image
    aligned_image = np.zeros([height, width, 3])
    aligned_image[:shape[0], :shape[1]] = crop
    aligned_image = aligned_image.astype(np.float32)

    # align mask
    aligned_msk = np.zeros([height, width])
    aligned_msk[:shape[0], :shape[1]] = crop_msk
    aligned_msk = (aligned_msk == 1).astype(np.uint8)

    # revise the intrinsic camera matrix
    K = K.copy()
    K[0, 2] = K[0, 2] - bbox[0, 0]
    K[1, 2] = K[1, 2] - bbox[0, 1]
    K = K.astype(np.float32)

    return aligned_image, aligned_msk, K, bbox


def random_crop_image(img, msk, K, min_size=80, max_size=88):
    H, W = img.shape[:2]
    min_HW = min(H, W)
    min_HW = min(min_HW, max_size)

    max_size = min_HW
    min_size = int(min(min_size, 0.8 * min_HW))
    H_size = np.random.randint(min_size, max_size)
    W_size = H_size
    x = 8
    H_size = (H_size | (x - 1)) + 1
    W_size = (W_size | (x - 1)) + 1

    # randomly select begin_x and begin_y
    coord = np.argwhere(msk == 1)
    center_xy = coord[np.random.randint(0, len(coord))][[1, 0]]
    min_x, min_y = center_xy[0] - W_size // 2, center_xy[1] - H_size // 2
    max_x, max_y = min_x + W_size, min_y + H_size
    if min_x < 0:
        min_x, max_x = 0, W_size
    if max_x > W:
        min_x, max_x = W - W_size, W
    if min_y < 0:
        min_y, max_y = 0, H_size
    if max_y > H:
        min_y, max_y = H - H_size, H

    # crop image and mask
    begin_x, begin_y = min_x, min_y
    img = img[begin_y:begin_y + H_size, begin_x:begin_x + W_size]
    msk = msk[begin_y:begin_y + H_size, begin_x:begin_x + W_size]

    # revise the intrinsic camera matrix
    K = K.copy()
    K[0, 2] = K[0, 2] - begin_x
    K[1, 2] = K[1, 2] - begin_y
    K = K.astype(np.float32)

    return img, msk, K


def get_bounds(xyz):
    min_xyz = np.min(xyz, axis=0)
    max_xyz = np.max(xyz, axis=0)
    min_xyz -= cfg.box_padding
    max_xyz += cfg.box_padding
    bounds = np.stack([min_xyz, max_xyz], axis=0)
    bounds = bounds.astype(np.float32)#[2,3]
    # visualize_pointcloud(xyz, bounds)
    return bounds

def visualize_pointcloud(pts, bounds):
    import numpy as np
    import torch
    import cv2
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import os
    import matplotlib.patches as patches
    # from sklearn.manifold import TSNE
    import open3d as o3d
    def create_coordinate_frame(size=1, origin=[0, 0, 0]):
        """创建一个Open3D坐标系对象"""
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=origin)
        return coord_frame

    point_cloud = pts
    # 将NumPy数组转换为Open3D的PointCloud对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)#.cpu().numpy())

    # 创建坐标轴对象
    coord_frame = create_coordinate_frame()

    # 创建bounding box
    aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound=bounds[0, :], max_bound=bounds[1, :])

    # 创建LineSet对象，并设置颜色为红色
    aabb.color = (0,0,0)

    # 将坐标轴对象添加到可视化窗口
    o3d.visualization.draw_geometries([pcd, coord_frame, aabb])


def prepare_sp_input(xyz):
    # obtain the bounds for coord construction
    bounds = get_bounds(xyz)
    # construct the coordinate
    dhw = xyz[:, [2, 1, 0]]
    min_dhw = bounds[0, [2, 1, 0]]
    max_dhw = bounds[1, [2, 1, 0]]
    voxel_size = np.array(cfg.voxel_size)
    coord = np.round((dhw - min_dhw) / voxel_size).astype(np.int32)
    # construct the output shape
    out_sh = np.ceil((max_dhw - min_dhw) / voxel_size).astype(np.int32)
    x = 32
    out_sh = (out_sh | (x - 1)) + 1
    return coord, out_sh, bounds


def crop_mask_edge(msk):
    msk = msk.copy()
    border = 10
    kernel = np.ones((border, border), np.uint8)
    msk_erode = cv2.erode(msk.copy(), kernel)
    msk_dilate = cv2.dilate(msk.copy(), kernel)
    msk[(msk_dilate - msk_erode) == 1] = 100
    return msk


def adjust_hsv(img, saturation, brightness, contrast):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv = hsv.astype(np.float32)
    hsv[..., 1] = hsv[..., 1] * saturation
    hsv[..., 1] = np.minimum(hsv[..., 1], 255)
    hsv[..., 2] = hsv[..., 2] * brightness
    hsv[..., 2] = np.minimum(hsv[..., 2], 255)
    hsv = hsv.astype(np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    img = img.astype(np.float32) * contrast
    img = np.minimum(img, 255)
    img = img.astype(np.uint8)
    return img


def transform_can_smpl(xyz):
    center = np.array([0, 0, 0]).astype(np.float32)
    rot = np.array([[np.cos(0), -np.sin(0)], [np.sin(0), np.cos(0)]])
    rot = rot.astype(np.float32)
    trans = np.array([0, 0, 0]).astype(np.float32)
    if np.random.uniform() > cfg.rot_ratio:
        return xyz, center, rot, trans

    xyz = xyz.copy()

    # rotate the smpl
    rot_range = np.pi / 32
    t = np.random.uniform(-rot_range, rot_range)
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    rot = rot.astype(np.float32)
    center = np.mean(xyz, axis=0)
    xyz = xyz - center
    xyz[:, [0, 2]] = np.dot(xyz[:, [0, 2]], rot.T)
    xyz = xyz + center

    # translate the smpl
    x_range = 0.05
    z_range = 0.025
    x_trans = np.random.uniform(-x_range, x_range)
    z_trans = np.random.uniform(-z_range, z_range)
    trans = np.array([x_trans, 0, z_trans]).astype(np.float32)
    xyz = xyz + trans

    return xyz, center, rot, trans

# main
if __name__ == '__main__':
    ray_sampler = FlexGridRaySampler(N_samples=16,
                                     min_scale=0.125,
                                     max_scale=1.,
                                     scale_anneal=0.0025 )
    iterations = 200*500
    k_iter = iterations // 1000 * 3
    min_scale = max(0.125, 1. * exp(-k_iter*0.01))
    min_scale = min(0.9, min_scale)

    ray_sampler.max_scale = min_scale
    output = ray_sampler.sample_rays(512,512)
    print(output)
