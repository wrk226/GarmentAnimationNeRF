import torch
from lib.config import cfg
from .nerf_net_utils import *
from .. import embedder
import numpy as np
import mcubes
import trimesh
from lib.utils.blend_utils import *
from tqdm import tqdm

class Renderer:
    def __init__(self, net, network_upper, network_lower):
        self.net = net

    def batchify_rays(self, wpts, alpha_decoder, net, chunk, batch):
        """Render rays in smaller minibatches to avoid OOM.
        """
        n_point = wpts.shape[0]
        all_ret = []
        for i in tqdm(range(0, n_point, chunk)):
            ret = alpha_decoder(wpts[i:i + chunk])
            all_ret.append(ret.detach().cpu().numpy())
        all_ret = np.concatenate(all_ret, 0)
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

    def render(self, batch):
        pts = torch.from_numpy(self.get_pts(batch)).cuda()
        sh = pts.shape
        pts = pts.reshape((-1, 3))

        alpha_decoder = lambda x: self.net.get_alpha(x, batch)

        alpha = self.batchify_rays(pts, alpha_decoder, self.net, 2048 * 64, batch)

        cube = alpha
        cube = cube.reshape(*sh[0:-1])
        cube = np.pad(cube, 10, mode='constant')
        vertices, triangles = mcubes.marching_cubes(cube, cfg.mesh_th)
        vertices = (vertices - 10) * cfg.voxel_size[0]
        vertices = vertices + batch['wbounds'][0, 0].detach().cpu().numpy()

        # mesh = trimesh.Trimesh(vertices, triangles)
        # labels = trimesh.graph.connected_component_labels(mesh.face_adjacency)
        # triangles = triangles[labels == 0]
        # import open3d as o3d
        # mesh_o3d = o3d.geometry.TriangleMesh()
        # mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
        # mesh_o3d.triangles = o3d.utility.Vector3iVector(triangles)
        # mesh_o3d.remove_unreferenced_vertices()
        # vertices = np.array(mesh_o3d.vertices)
        # triangles = np.array(mesh_o3d.triangles)

        ret = {
            'posed_vertex': vertices,
            'triangle': triangles,
            # 'rgb': rgb,
        }

        return ret
