import torch
from lib.config import cfg
from .nerf_net_utils import *
from .. import embedder
import numpy as np
import mcubes
import trimesh
from lib.utils.blend_utils import *
from lib.utils import sample_utils
from tqdm import tqdm
from lib.utils import net_utils, sdf_utils
class Renderer:
    def __init__(self, net, net_upper_body=None, net_lower_body=None):
        self.net = net
        self.clothes_change = False
        if net_upper_body or net_lower_body:
            self.clothes_change = True
            self.net_upper = net_upper_body if net_upper_body is not None else net
            self.net_lower = net_lower_body if net_lower_body is not None else net

    def batchify_rays(self, wpts, net, chunk, batch, get_alpha=False):
        """Render rays in smaller minibatches to avoid OOM.
        """
        n_point = wpts.shape[0]
        all_sdf = []
        all_inv_variance = []
        for i in tqdm(range(0, n_point, chunk)):
            # resd = net.calculate_residual_deformation(wpts[i:i + chunk][None],
            #                                           batch['latent_index'])
            # wpts[i:i + chunk] = wpts[i:i + chunk] + resd[0]
            sdf_chunk = self.net.sdf_decoder(torch.from_numpy(wpts[i:i + chunk]).cuda(), batch)#[131072, 1]

            all_sdf.append(sdf_chunk.detach().cpu().numpy())

        all_sdf = np.concatenate(all_sdf, 0)
        if get_alpha:
            all_inv_variance = np.concatenate(all_inv_variance, 0)
            alpha = sdf_utils.sdf_to_alpha(torch.from_numpy(all_sdf), torch.from_numpy(all_inv_variance), None, None,
                                   n_point, batch).numpy()
            return alpha
        return all_sdf

    def batchify_blend_weights(self, pts, bw_input, chunk=1024 * 32):
        all_ret = []
        for i in range(0, pts.shape[1], chunk):
            ret = self.net.calculate_bigpose_smpl_bw(pts[:, i:i + chunk],
                                                     bw_input)
            all_ret.append(ret)
        all_ret = torch.cat(all_ret, 2)
        return all_ret

    def batchify_normal_sdf(self, pts, batch, chunk=1024 * 32):
        all_normal = []
        all_sdf = []
        for i in range(0, pts.shape[1], chunk):
            normal, sdf = self.net.gradient_of_deformed_sdf(
                pts[:, i:i + chunk], batch)
            all_normal.append(normal.detach().cpu().numpy())
            all_sdf.append(sdf.detach().cpu().numpy())
        all_normal = np.concatenate(all_normal, axis=1)
        all_sdf = np.concatenate(all_sdf, axis=1)
        return all_normal, all_sdf

    def batchify_residual_deformation(self, wpts, batch, chunk=1024 * 32):
        all_ret = []
        for i in range(0, wpts.shape[1], chunk):
            ret = self.net.calculate_residual_deformation(wpts[:, i:i + chunk],
                                                          batch)
            all_ret.append(ret)
        all_ret = torch.cat(all_ret, 1)
        return all_ret

    def get_pts(self, batch):
        wbounds = batch['wbounds'][0].clone().cpu()

        voxel_size = cfg.voxel_size
        x = np.arange(wbounds[0, 0], wbounds[1, 0] + voxel_size[0],
                      voxel_size[0])
        y = np.arange(wbounds[0, 1], wbounds[1, 1] + voxel_size[1],
                      voxel_size[1])
        z = np.arange(wbounds[0, 2], wbounds[1, 2] + voxel_size[2],
                      voxel_size[2])
        pts = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1)
        pts = pts.astype(np.float32)
        pts = pts
        return pts

    def render(self, batch, get_alpha=False):

        pts = self.get_pts(batch)
        sh = pts.shape
        pts = pts.reshape((-1, 3))

        # 使用smpl模型做mask
        # tbw, tnorm = sample_utils.sample_blend_closest_points(pts, batch['tvertices'], batch['weights'])
        # tnorm = tnorm[..., 0]
        # norm_th = 0.1
        # inside = tnorm < norm_th
        #
        # pts = pts[inside]
        #1:13
        sdf = self.batchify_rays(pts, self.net, 2048*64 , batch)#2048* 64
        # todo: test
        # sdf = -sdf[:, 0]
        # sdf = sdf[:, 0]
        # if cfg.use_udf:

        cube = sdf.reshape(*sh[0:-1])
        # cube = np.pad(cube, 10, mode='constant', constant_values=10)
        udf_min = np.min(sdf)
        udf_max = np.max(sdf)
        th = 0.

        if cfg.use_udf and (th < udf_min or th > udf_max):
            th = 0.99*udf_min+0.01*udf_max
        else:
            th = th

        # th=1.
        # print(th)
        # print("=======================")
        from skimage.measure import marching_cubes
        # vertices, triangles, _, _ = marching_cubes(cube, th)
        vertices, triangles = mcubes.marching_cubes(cube, th)# 0 for sdf, 50 for nerf
        mesh = trimesh.Trimesh(vertices, triangles)
        if len(mesh.vertices) == 0:
            return None
        # mesh = max(mesh.split(), key=lambda m: len(m.vertices))
        vertices, triangles = mesh.vertices, mesh.faces
        vertices = vertices  * cfg.voxel_size[0]
        vertices = vertices + batch['wbounds'][0, 0].detach().cpu().numpy()

        ret = {
            # 'vertex': vertices,
            'posed_vertex': vertices,
            'triangle': triangles,
        }

        return ret
