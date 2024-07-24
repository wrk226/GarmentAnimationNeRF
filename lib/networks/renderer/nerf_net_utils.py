import torch.nn.functional as F
import torch
from lib.config import cfg

# from FFJORD github code
def divergence_approx(input_points, offsets_of_inputs):  # , as_loss=True):
    # avoids explicitly computing the Jacobian
    e = torch.randn_like(offsets_of_inputs, device=offsets_of_inputs.get_device())
    e_dydx = torch.autograd.grad(offsets_of_inputs, input_points, e, create_graph=True,
                                 # retain_graph=True, only_inputs=True
                                 )[0]
    e_dydx_e = e_dydx * e
    approx_tr_dydx = e_dydx_e.view(offsets_of_inputs.shape[0], -1).sum(dim=1)
    return approx_tr_dydx

def raw2outputs_nerf(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) *
                                                                 dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat(
        [dists,
         torch.Tensor([1e10]).expand(dists[..., :1].shape).to(dists.device)],
        -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 3].shape) * raw_noise_std

    alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(
        torch.cat(
            [torch.ones((alpha.shape[0], 1), device=alpha.device), 1. - alpha + 1e-10],
            -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map, device=depth_map.device),
                              depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map

def raw2outputs_neus(raw, z_vals, white_bkgd=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    rgb = raw[..., :-1]  # [N_rays, N_samples, 3]
    alpha = raw[..., -1]

    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(
        torch.cat(
            [torch.ones((alpha.shape[0], 1), device=alpha.device), 1. - alpha + 1e-10],
            -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map, device=depth_map.device),
                              depth_map / (torch.sum(weights, -1)+1e-10))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])
    # if rgb_map.max()>10:
    #     print('rgb_map.max()>10')
    return rgb_map, disp_map, acc_map, weights, depth_map


# Hierarchical sampling (section 5.2)
# def sample_pdf(bins, weights, N_samples, det=False):
#     from torchsearchsorted import searchsorted
#
#     # Get pdf
#     weights = weights + 1e-5  # prevent nans
#     pdf = weights / torch.sum(weights, -1, keepdim=True)
#     cdf = torch.cumsum(pdf, -1)
#     cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))
#     # Take uniform samples
#     if det:
#         u = torch.linspace(0., 1., steps=N_samples, device=cdf.device)
#         u = u.expand(list(cdf.shape[:-1]) + [N_samples])
#     else:
#         u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=cdf.device)
#
#     # Invert CDF
#     u = u.contiguous()
#     inds = searchsorted(cdf, u, side='right')
#     below = torch.max(torch.zeros_like(inds - 1), inds - 1)
#     above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
#     inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)
#
#     # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
#     # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
#     matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
#     cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
#     bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)
#
#     denom = (cdf_g[..., 1] - cdf_g[..., 0])
#     denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
#     t = (u - cdf_g[..., 0]) / denom
#     samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
#
#     return samples


def get_intersection_mask(sdf, z_vals):
    """
    sdf: n_batch, n_pixel, n_sample
    z_vals: n_batch, n_pixel, n_sample
    """
    sign = torch.sign(sdf[..., :-1] * sdf[..., 1:])
    ind = torch.min(sign * torch.arange(sign.size(2)).flip([0]).to(sign),
                    dim=2)[1]
    sign = sign.min(dim=2)[0]
    intersection_mask = sign == -1
    return intersection_mask, ind


def sphere_tracing(wpts, sdf, z_vals, ray_o, ray_d, decoder):
    """
    wpts: n_point, n_sample, 3
    sdf: n_point, n_sample
    z_vals: n_point, n_sample
    ray_o: n_point, 3
    ray_d: n_point, 3
    """
    sign = torch.sign(sdf[..., :-1] * sdf[..., 1:])
    ind = torch.min(sign * torch.arange(sign.size(1)).flip([0]).to(sign),
                    dim=1)[1]

    wpts_sdf = sdf[torch.arange(len(ind)), ind]
    wpts_start = wpts[torch.arange(len(ind)), ind]
    wpts_end = wpts[torch.arange(len(ind)), ind + 1]

    sdf_threshold = 5e-5
    unfinished_mask = wpts_sdf.abs() > sdf_threshold
    i = 0
    while unfinished_mask.sum() != 0 and i < 20:
        curr_start = wpts_start[unfinished_mask]
        curr_end = wpts_end[unfinished_mask]

        wpts_mid = (curr_start + curr_end) / 2
        mid_sdf = decoder(wpts_mid)[:, 0]

        ind_outside = mid_sdf > 0
        if ind_outside.sum() > 0:
            curr_start[ind_outside] = wpts_mid[ind_outside]

        ind_inside = mid_sdf < 0
        if ind_inside.sum() > 0:
            curr_end[ind_inside] = wpts_mid[ind_inside]

        wpts_start[unfinished_mask] = curr_start
        wpts_end[unfinished_mask] = curr_end
        wpts_sdf[unfinished_mask] = mid_sdf
        unfinished_mask[unfinished_mask] = (mid_sdf.abs() >
                                            sdf_threshold) | (mid_sdf < 0)

        i = i + 1

    # get intersection points
    mask = (wpts_sdf.abs() < sdf_threshold) * (wpts_sdf >= 0)
    intersection_points = wpts_start[mask]

    ray_o = ray_o[mask]
    ray_d = ray_d[mask]
    z_vals = (intersection_points[:, 0] - ray_o[:, 0]) / ray_d[:, 0]

    return intersection_points, z_vals, mask

# def up_sample( rays_o, rays_d, z_vals, sdf, n_importance, inv_s):
#     """
#     Up sampling give a fixed inv_s
#     """
#     batch_size, n_samples = z_vals.shape
#     pts = rays_o[:, :, None] + rays_d[:, :, None] * z_vals[..., None]  # n_rays, n_samples, 3
#     sdf = sdf.reshape(batch_size, n_samples)
#
#     if cfg.use_udf:
#         cdf = (sdf * inv_s/(1+sdf * inv_s)).clip(0., 1e6)
#     else:
#         cdf = torch.sigmoid(sdf * inv_s)
#
#     residual = cdf[:, :-1] - cdf[:, 1:]
#     if cfg.use_udf:
#         p = torch.abs(residual)
#     else:
#         p = residual
#     c = cdf[:, 1:]
#
#     alpha = ((p + 1e-5) / (c + 1e-5)).clamp(0.0, 1.0)
#
#     weights = alpha * torch.cumprod(
#         torch.cat([torch.ones([batch_size, 1], device=alpha.device), 1. - alpha + 1e-7], -1), -1)[:, :-1]
#
#     z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
#     return z_samples
def up_sample( rays_o, rays_d, z_vals, sdf, n_importance, inv_s):
    """
    Up sampling give a fixed inv_s
    """
    batch_size, n_samples = z_vals.shape
    pts = rays_o[:, :, None] + rays_d[:, :, None] * z_vals[..., None]  # n_rays, n_samples, 3
    # todo:看看这个有没有影响
    sdf = sdf.reshape(batch_size, n_samples)

    prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
    prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
    mid_sdf = (prev_sdf + next_sdf) * 0.5
    # 计算cos值
    cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

    # ----------------------------------------------------------------------------------------------------------
    # Use min value of [ cos, prev_cos ]
    # Though it makes the sampling (not rendering) a little bit biased, this strategy can make the sampling more
    # robust when meeting situations like below:
    #
    # SDF
    # ^
    # |\          -----x----...
    # | \        /
    # |  x      x
    # |---\----/-------------> 0 level
    # |    \  /
    # |     \/
    # |
    # ----------------------------------------------------------------------------------------------------------
    # 用0作为第一个采样点的cos值。
    prev_cos_val = torch.cat([torch.zeros([batch_size, 1], device=cos_val.device), cos_val[:, :-1]], dim=-1)
    cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
    # Use min value of [ cos, prev_cos ]
    cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
    # 距离函数为正的点的cos值取0,不在圆内的都取0。
    cos_val = cos_val.clip(-1e3, 0.0)

    dist = (next_z_vals - prev_z_vals)
    if cfg.use_udf:
        prev_esti_sdf = (mid_sdf - cos_val * dist * 0.5).clip(0., 1e6)
        next_esti_sdf = (mid_sdf + cos_val * dist * 0.5).clip(0., 1e6)
        prev_cdf = prev_esti_sdf*inv_s/(1+prev_esti_sdf*inv_s)
        next_cdf = next_esti_sdf*inv_s/(1+next_esti_sdf*inv_s)
        alpha = ((torch.abs(prev_cdf - next_cdf) +0) / (torch.abs(prev_cdf) + 1e-20)).clip(0., 1.)
    else:
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)



    weights = alpha * torch.cumprod(
        torch.cat([torch.ones([batch_size, 1], device=alpha.device), 1. - alpha + 1e-7], -1), -1)[:, :-1]

    z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
    return z_samples
# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        # u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples)
        u = torch.linspace(0., 1., steps=n_samples, device=cdf.device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples], device=cdf.device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples