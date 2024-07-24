import torch

def rgb_to_hsv(input):
    tensor_shape = input.shape
    input = input.reshape(-1,3)
    # Assuming input is of shape [n_rays, 3]
    r, g, b = input[:, 0], input[:, 1], input[:, 2]

    c_max, _ = torch.max(input, dim=1)
    c_min, _ = torch.min(input, dim=1)
    diff = c_max - c_min

    h = torch.zeros_like(c_max)
    h[diff != 0] = torch.where(c_max[diff != 0] == r[diff != 0], ((60 * (g[diff != 0] - b[diff != 0]) / diff[diff != 0]) + 360) % 360, h[diff != 0])
    h[diff != 0] = torch.where(c_max[diff != 0] == g[diff != 0], (60 * (b[diff != 0] - r[diff != 0]) / diff[diff != 0]) + 120, h[diff != 0])
    h[diff != 0] = torch.where(c_max[diff != 0] == b[diff != 0], (60 * (r[diff != 0] - g[diff != 0]) / diff[diff != 0]) + 240, h[diff != 0])

    s = torch.zeros_like(c_max)
    s[c_max != 0] = (diff[c_max != 0] / c_max[c_max != 0]) * 100

    v = c_max * 100

    output = torch.stack((h, s, v), dim=1)
    return output.reshape(*tensor_shape)

def hsv_to_rgb(input):
    tensor_shape = input.shape
    input = input.reshape(-1,3)
    # Assuming input is of shape [n_rays, 3]
    h, s, v = input[:, 0], input[:, 1], input[:, 2]

    c = (s / 100) * (v / 100)
    x = c * (1 - torch.abs((h / 60) % 2 - 1))
    m = v / 100 - c

    r, g, b = torch.zeros_like(h), torch.zeros_like(h), torch.zeros_like(h)

    idx = (h >= 0) & (h < 60)
    r[idx], g[idx] = c[idx], x[idx]

    idx = (h >= 60) & (h < 120)
    r[idx], g[idx] = x[idx], c[idx]

    idx = (h >= 120) & (h < 180)
    g[idx], b[idx] = c[idx], x[idx]

    idx = (h >= 180) & (h < 240)
    g[idx], b[idx] = x[idx], c[idx]

    idx = (h >= 240) & (h < 300)
    r[idx], b[idx] = x[idx], c[idx]

    idx = (h >= 300) & (h < 360)
    r[idx], b[idx] = c[idx], x[idx]

    output = torch.stack((r + m, g + m, b + m), dim=1)
    return output.reshape(*tensor_shape)

def calculate_delta_hsv(basis_rgb_orig, basis_rgb_new, num_basis=2):
    delta_hsv = torch.zeros_like(basis_rgb_new)
    delta_hsv[...,1:3] = 1.

    rgb_all = torch.cat([basis_rgb_orig, basis_rgb_new], dim=0)
    hsv_all = rgb_to_hsv(rgb_all)
    hsv_orig = hsv_all[:num_basis]
    hsv_new = hsv_all[num_basis:]

    delta_hsv[:, 0] = torch.fmod((hsv_new[:,0]-hsv_orig[:,0]+360), 360)
    delta_hsv[:, 1] = (hsv_new[:,1]/(hsv_orig[:,1]+1e-9))
    delta_hsv[:, 2] = (hsv_new[:,2]/(hsv_orig[:,2]+1e-9))
    return delta_hsv

def get_deformed_rgb(rgb_orig, delta_hsv):
    hsv = rgb_to_hsv(rgb_orig)
    hsv_new = hsv.clone()
    hsv_new[...,0] = torch.fmod((hsv[...,0]+delta_hsv[...,0]+360), 360)
    hsv_new[...,1] = torch.clip((hsv[...,1]*delta_hsv[...,1]), 0,100)
    hsv_new[...,2] = torch.clip((hsv[...,2]*delta_hsv[...,2]), 0,100)
    rgb_deformed = hsv_to_rgb(hsv_new)
    return rgb_deformed