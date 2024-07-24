import os
import sys
from lib.config import cfg
import numpy as np
import trimesh
import pickle
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pytorch3d
from pytorch3d.structures import Meshes, Pointclouds
from lib.utils import vis_utils
from .map_utils import load_model, point_mesh_face_distance, barycentric_to_points, points_to_barycentric
from .uv_map_generator import UV_Map_Generator
# import time
# import psutil
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
import time
from lib.utils import vis_utils,img_utils
# from kaolin.ops.mesh import check_sign
# from kaolin.metrics.trianglemesh import point_to_mesh_distance

class SurfaceAlignedConverter:

    def __init__(self):
        # load body template
        obj_path = os.path.join(cfg.train_dataset.data_root,f'{cfg.body_template}.obj')
        verts, faces, faces_idx, verts_uvs, faces_t, faces_n, device = load_model(obj_path)
        self.device = device
        self.faces_idx = faces_idx # v
        self.verts_uvs = verts_uvs # [7576, 2] --> previous [6890, 2] equals to verts' idx
        self.faces_t = faces_t # vt
        self.faces_n = faces_n # vn
        verts = verts[:, [0, 2, 1]]
        verts[:, 1] *= -1
        self.verts = verts
        self.load_mesh_topology(verts, faces_idx, os.path.dirname(obj_path))
        self.load_uv_map(obj_path, os.path.dirname(obj_path), prefix='b_')



    def _parse_obj(self, obj_file):
        with open(obj_file, 'r') as fin:
            lines = [l
                for l in fin.readlines()
                if len(l.split()) > 0
                and not l.startswith('#')
            ]

        # Load all vertices (v) and texcoords (vt)
        vertices = []
        texcoords = []

        for line in lines:
            lsp = line.split()
            if lsp[0] == 'v':
                x = float(lsp[1])
                y = float(lsp[2])
                z = float(lsp[3])
                vertices.append((x, y, z))
            elif lsp[0] == 'vt':
                u = float(lsp[1])
                v = float(lsp[2])
                texcoords.append((1 - v, u))

        # Stack these into an array
        vertices = np.vstack(vertices).astype(np.float32)
        texcoords = np.vstack(texcoords).astype(np.float32)

        # Load face data. All lines are of the form:
        # f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3
        #
        # Store the texcoord faces and a mapping from texcoord faces
        # to vertex faces
        vt_faces = []
        faces = []
        vt_to_v = {}
        v_to_vt = [None] * vertices.shape[0]
        for i in range(vertices.shape[0]):
            v_to_vt[i] = set()

        for line in lines:
            vs = line.split()
            if vs[0] == 'f':
                v0 = int(vs[1].split('/')[0]) - 1
                v1 = int(vs[2].split('/')[0]) - 1
                v2 = int(vs[3].split('/')[0]) - 1
                vt0 = int(vs[1].split('/')[1]) - 1
                vt1 = int(vs[2].split('/')[1]) - 1
                vt2 = int(vs[3].split('/')[1]) - 1
                faces.append((v0, v1, v2))
                vt_faces.append((vt0, vt1, vt2))
                vt_to_v[vt0] = v0
                vt_to_v[vt1] = v1
                vt_to_v[vt2] = v2
                v_to_vt[v0].add(vt0)
                v_to_vt[v1].add(vt1)
                v_to_vt[v2].add(vt2)

        vt_faces = np.vstack(vt_faces)
        faces = np.vstack(faces)
        tmp_dict = {
            'vertices': vertices, # 顶点坐标
            'texcoords': texcoords, # uv坐标
            'vt_faces': vt_faces, # uv面
            'faces': faces, # 面
            'vt_to_v': vt_to_v, # uv顶点序号->顶点序号
            'v_to_vt': v_to_vt # 顶点序号->uv顶点序号
        }
        return tmp_dict


    def load_uv_map(self, obj_path, cache_path='zju_smpl/cache/', prefix='b_'):
        suffix = '_coarse_garment'
        uv_info_path = os.path.join(cache_path, f'{prefix}uv_info_{cfg.uv_size}{suffix}.pickle')
        # 改成缓存形式
        if not os.path.exists(uv_info_path):
            print('==> uv info cache not found, start calculating...(This could take a few minutes)', cache_path)
            data_root = os.path.dirname(obj_path)
            file_name = os.path.basename(obj_path).split(".")[0]
            generator = UV_Map_Generator(
                UV_height=cfg.uv_size,
                data_root=data_root,
                file_name=file_name,
            )
            generator.render_UV_atlas(f'{data_root}/{prefix}{file_name}_atlas_{cfg.uv_size}{suffix}.png')
            ###############################################
            img, verts, rgbs = generator.render_point_cloud(f'{data_root}/{prefix}{file_name}_{cfg.uv_size}{suffix}.png')
            verts, rgbs = generator.write_ply(f'{data_root}/{prefix}{file_name}_{cfg.uv_size}{suffix}.ply', verts, rgbs)
            uv, _ = generator.get_UV_map(verts, dilate=False)
            uv = uv.max(axis=2)
            binary_mask = np.where(uv > 0, 1., 0.)
            binary_mask = (binary_mask * 255).astype(np.uint8)
            imsave(f'{data_root}/{prefix}{file_name}_UV_mask{suffix}.png', binary_mask)
            #################################################
            uv_info = {
                'pixel_bary_weight': generator.bary_weights,
                'pixel_face': generator.face_id,
                'uv_faces': generator.vt_faces, # [13776, 3]
                'uv_verts': generator.texcoords, # [7576, 2]
                'uv_triangles': generator.texcoords[generator.vt_faces], # [13776, 3, 2]
            }

            with open(uv_info_path, 'wb') as wf:
                pickle.dump(uv_info, wf)
            print('==> Finished! Cache saved to: ', cache_path)
        else:
            print('==> Find pre-computed uv info! Loading cache from: ', cache_path)
        with open(uv_info_path, 'rb') as rf:
            bary_info = pickle.load(rf)
            if prefix == 'b_':
                self.pixel_bary_weight = torch.from_numpy(bary_info['pixel_bary_weight']).to(self.device)
                self.pixel_face = torch.from_numpy(bary_info['pixel_face']).to(self.device)
                self.uv_faces = torch.from_numpy(bary_info['uv_faces']).to(self.device)
                self.uv_verts = torch.from_numpy(bary_info['uv_verts']).to(self.device)
                self.uv_triangles = torch.from_numpy(bary_info['uv_triangles']).to(self.device)
            elif prefix == 'g_':
                self.g_pixel_bary_weight = torch.from_numpy(bary_info['pixel_bary_weight']).to(self.device)
                self.g_pixel_face = torch.from_numpy(bary_info['pixel_face']).to(self.device)
                self.g_uv_faces = torch.from_numpy(bary_info['uv_faces']).to(self.device)
                self.g_uv_verts = torch.from_numpy(bary_info['uv_verts']).to(self.device)
                self.g_uv_triangles = torch.from_numpy(bary_info['uv_triangles']).to(self.device)


    def get_ref_weights(self, verts, faces, camera_location, garment=False):
        if garment:
            faces_idx = self.g_faces_idx
            pixel_face = self.g_pixel_face
            pixel_bary_weight = self.g_pixel_bary_weight
        else:
            faces_idx = self.faces_idx
            pixel_face = self.pixel_face
            pixel_bary_weight = self.pixel_bary_weight
        meshes = Meshes(verts=verts, faces=faces)
        # w_normal_uv = self.converter.extract_verts_normal_info(batch['wverts'])
        verts_normal = meshes.verts_normals_packed()

        d_vert2cam = torch.nn.functional.normalize(camera_location - verts[0], p=2, dim=1)
        verts_normal = torch.nn.functional.normalize(verts_normal, p=2, dim=1)
        cos_similarity = torch.sum(d_vert2cam * verts_normal, dim=1, keepdim=True).float()
        pixel2face = pixel_face # [256,256]
        pixel_to_tri_verts = faces_idx[pixel2face.long()] # [256,256,3]
        pixel_to_tri_cos = cos_similarity[pixel_to_tri_verts] # [256,256,3,3]
        cos_sim_uv = torch.einsum('bijk,bij->bik', pixel_to_tri_cos, pixel_bary_weight) # [256,256,3]


        cos_sim_uv = cos_sim_uv.squeeze()
        return cos_sim_uv

    def extract_verts_normal_info(self, verts, garment=False):
        if garment:
            faces_idx = self.g_faces_idx
            pixel_face = self.g_pixel_face
            pixel_bary_weight = self.g_pixel_bary_weight
        else:
            faces_idx = self.faces_idx
            pixel_face = self.pixel_face
            pixel_bary_weight = self.pixel_bary_weight
        faces = faces_idx[None, ...].repeat(len(verts), 1, 1)

        meshes = Meshes(verts=verts, faces=faces)
        # 每个面
        verts_normals = meshes.verts_normals_packed() # [6890, 3]
        pixel2face = pixel_face # [256,256]
        pixel_to_tri_verts = faces_idx[pixel2face.long()] # [256,256,3]
        pixel_to_tri_normals = verts_normals[pixel_to_tri_verts] # [256,256,3,3]
        normal_uv = torch.einsum('bijk,bij->bik', pixel_to_tri_normals, pixel_bary_weight) # [256,256,3]


        normal_uv = normal_uv[None].permute(0, 3, 1, 2)
        return normal_uv


    def extract_verts_velocity_info_v2(self, velocity, garment=False):
        if garment:
            faces_idx = self.g_faces_idx
            pixel_face = self.g_pixel_face
            pixel_bary_weight = self.g_pixel_bary_weight
        else:
            faces_idx = self.faces_idx
            pixel_face = self.pixel_face
            pixel_bary_weight = self.pixel_bary_weight

        # verts_next = [verts] + verts_prev[:-1]
        # verts_prev = torch.cat(verts_prev, dim=0)
        # verts_next = torch.cat(verts_next, dim=0)
        velocity_scale = 5.
        # velocity = (verts_next - verts_prev) / (cfg.frame_interval * 1.) # [10, 6890, 3]
        velocity = torch.cat(velocity, dim=0).squeeze(1)
        pixel2face = pixel_face.long() # [256, 256]
        pixel_to_tri_verts = faces_idx[pixel2face] # [256, 256, 3]
        # todo:需要替换成batch操作， 删除循环
        velocity_uv = []
        for idx, vel in enumerate(velocity):
            pixel2velocitys = vel[pixel_to_tri_verts] # [n, 256,256,3,3]
            vel_uv = torch.einsum('bijk,bij->bik', pixel2velocitys, pixel_bary_weight) # [n, 256,256,3]
            ################visualize######################
            # uv_mask = self.pixel_face != -1
            # velocity_plt = vel_uv.clone().detach()
            # velocity_plt[uv_mask] = (velocity_plt[uv_mask]*velocity_scale+1)/2.
            # velocity_plt[~uv_mask] = 1
            # remove plt axis
            # plt.axis('off')
            # plt.imsave('velocity_{}.png'.format(221-idx), velocity_plt.detach().cpu().numpy())
            # plt.imshow(velocity_plt.detach().cpu().numpy())
            # plt.show()
            # plt.subplot(5,5, 2+idx)
            # plt.imshow(velocity_plt.detach().cpu().numpy())
            ###############################################
            vel_uv = vel_uv[None].permute(0, 3, 1, 2) * velocity_scale
            velocity_uv.append(vel_uv)

        if cfg.descripter_encoder_type== 'spade':
            velocity_uv = [torch.cat(velocity_uv[1:], dim=1), torch.cat(velocity_uv[:-1], dim=1)]
        else:
            velocity_uv = torch.cat(velocity_uv, dim=1)
        if cfg.sdf_input.loc:
            pixel_to_tri_xyz = verts.squeeze()[pixel_to_tri_verts] # [256,256,3,3]
            xyz_uv = torch.einsum('bijk,bij->bik', pixel_to_tri_xyz, pixel_bary_weight) # [256,256,3]

            xyz_uv = xyz_uv[None].permute(0, 3, 1, 2)
            return velocity_uv, xyz_uv
        # plt.show()
        return velocity_uv

    def extract_verts_velocity_info(self, verts, verts_prev=None, garment=False):
        if garment:
            faces_idx = self.g_faces_idx
            pixel_face = self.g_pixel_face
            pixel_bary_weight = self.g_pixel_bary_weight
        else:
            faces_idx = self.faces_idx
            pixel_face = self.pixel_face
            pixel_bary_weight = self.pixel_bary_weight

        verts_next = [verts] + verts_prev[:-1]
        verts_prev = torch.cat(verts_prev, dim=0)
        verts_next = torch.cat(verts_next, dim=0)
        velocity_scale = 5.
        velocity = (verts_next - verts_prev) / (cfg.frame_interval * 1.) # [10, 6890, 3]
        pixel2face = pixel_face.long() # [256, 256]
        pixel_to_tri_verts = faces_idx[pixel2face] # [256, 256, 3]
        # todo:需要替换成batch操作， 删除循环
        velocity_uv = []
        for idx, vel in enumerate(velocity):
            pixel2velocitys = vel[pixel_to_tri_verts] # [n, 256,256,3,3]
            vel_uv = torch.einsum('bijk,bij->bik', pixel2velocitys, pixel_bary_weight) # [n, 256,256,3]

            vel_uv = vel_uv[None].permute(0, 3, 1, 2) * velocity_scale
            velocity_uv.append(vel_uv)

        if cfg.descripter_encoder_type== 'spade':
            velocity_uv = [torch.cat(velocity_uv[1:], dim=1), torch.cat(velocity_uv[:-1], dim=1)]
        else:
            velocity_uv = torch.cat(velocity_uv, dim=1)
        if cfg.sdf_input.loc:
            pixel_to_tri_xyz = verts.squeeze()[pixel_to_tri_verts] # [256,256,3,3]
            xyz_uv = torch.einsum('bijk,bij->bik', pixel_to_tri_xyz, pixel_bary_weight) # [256,256,3]

            xyz_uv = xyz_uv[None].permute(0, 3, 1, 2)
            return velocity_uv, xyz_uv
        # plt.show()
        return velocity_uv



    def load_mesh_topology(self, verts, faces_idx, cache_path='zju_smpl/cache', prefix='b_'):
        suffix = '_gt' if cfg.gt_vertices else '_pred'
        if not os.path.exists(os.path.join(cache_path, f'{prefix}faces_to_corres_edges{suffix}.npy')):
            print('==> Computing mesh topology... ', cache_path)
            faces_to_corres_edges, edges_to_corres_faces, verts_to_corres_faces, edges_to_corres_verts = self._parse_mesh(verts, faces_idx)
            # save cache
            os.makedirs(cache_path, exist_ok=True)
            np.save(os.path.join(cache_path, f'{prefix}faces_to_corres_edges{suffix}.npy'), faces_to_corres_edges.to('cpu').detach().numpy().copy())
            np.save(os.path.join(cache_path, f'{prefix}edges_to_corres_faces{suffix}.npy'), edges_to_corres_faces.to('cpu').detach().numpy().copy())
            np.save(os.path.join(cache_path, f'{prefix}verts_to_corres_faces{suffix}.npy'), verts_to_corres_faces.to('cpu').detach().numpy().copy())
            np.save(os.path.join(cache_path, f'{prefix}edges_to_corres_verts{suffix}.npy'), edges_to_corres_verts.to('cpu').detach().numpy().copy())

            print('==> Finished! Cache saved to: ', cache_path)

        else:
            print('==> Find pre-computed mesh topology! Loading cache from: ', cache_path)
            faces_to_corres_edges = torch.from_numpy(np.load(os.path.join(cache_path, f'{prefix}faces_to_corres_edges{suffix}.npy')))
            edges_to_corres_faces = torch.from_numpy(np.load(os.path.join(cache_path, f'{prefix}edges_to_corres_faces{suffix}.npy')))
            verts_to_corres_faces = torch.from_numpy(np.load(os.path.join(cache_path, f'{prefix}verts_to_corres_faces{suffix}.npy')))
            edges_to_corres_verts = torch.from_numpy(np.load(os.path.join(cache_path, f'{prefix}edges_to_corres_verts{suffix}.npy')))

        self.faces_to_corres_edges = faces_to_corres_edges.long().to(self.device)  # [13776, 3]
        self.edges_to_corres_faces = edges_to_corres_faces.long().to(self.device)  # [20664, 2]
        self.verts_to_corres_faces = verts_to_corres_faces.long().to(self.device)  # [6890, 9]
        self.edges_to_corres_verts = edges_to_corres_verts.long().to(self.device)  # [20664, 2]


    def xyz_to_xyzch_sr(self, descripter=None, batch=None, garment=False,sr_info=None):
        if garment:
            uv_verts = self.g_uv_verts
            uv_faces = self.g_uv_faces
        else:
            uv_verts = self.uv_verts
            uv_faces = self.uv_faces


        sr_bary_coords = sr_info['bary_coords']
        valid_mask = torch.sum(sr_bary_coords,dim=-1)>0
        valid_bc = sr_bary_coords[valid_mask]

        sr_pix_to_face = sr_info['pix_to_face']
        valid_faces = sr_pix_to_face[valid_mask]


        xyc_triangles = uv_verts[uv_faces.long()][valid_faces] #[32768,3,2]
        xyc = barycentric_to_points(xyc_triangles, valid_bc) # [batch*65536, 2], 采样点在uv上的坐标
        descripter_rs = descripter.clone()[0].permute(1, 2, 0)
        descripter_sample = img_utils.get_colors_from_positions(xyc*(cfg.uv_size-1), descripter_rs)
        nr_feat = torch.zeros(cfg.unet_dim,cfg.unet_dim,descripter_sample.shape[-1], device='cuda')
        nr_feat[valid_mask] = descripter_sample
        return nr_feat



    def xyz_to_xyzch(self, points, verts, batch=None, garment=False, sr_info=None):
        if garment:
            self_verts = self.g_verts
            faces_idx = self.g_faces_idx
            verts_uvs = self.g_verts_uvs
            faces_t = self.g_faces_t
        else:
            self_verts = self.verts
            faces_idx = self.faces_idx
            verts_uvs = self.verts_uvs
            faces_t = self.faces_t

        h, barycentric, proj_face_idx, local_coordinates, nearest_new = self.projection(points, verts, revise=cfg.proj_mode == 'sparse',
                                                                                        batch=batch, faces_idx=faces_idx,verts_uvs=verts_uvs,
                                                                                        faces_t=faces_t, garment=garment)
        xyzc_triangles = self_verts[faces_idx].repeat(len(verts), 1, 1)[proj_face_idx.view(-1)]  # [batch*65536, 3, 3]
        cano_proj = barycentric_to_points(xyzc_triangles, barycentric.view(-1, 3)) # [batch*65536, 3], 采样点在mesh上的坐标

        ret = {'h': h.squeeze(),
               'barycentric': barycentric,
               'proj_face_idx': proj_face_idx,
               'local_coordinates': local_coordinates,
               'nearest_new': nearest_new,
               'cano_proj': cano_proj,
               }
        return ret

    def get_descriptor_for_each_point(self, descriptor_uv, proj_info, batch, garment=False):
        if garment:
            uv_verts = self.g_uv_verts
            uv_faces = self.g_uv_faces
        else:
            uv_verts = self.uv_verts
            uv_faces = self.uv_faces
        proj_face_idx = proj_info['proj_face_idx']
        barycentric = proj_info['barycentric']
        xyc_triangles = uv_verts[uv_faces.long()][proj_face_idx.view(-1)] #[32768,3,2]
        xyc = barycentric_to_points(xyc_triangles, barycentric.view(-1, 3)) # [batch*65536, 2], 采样点在uv上的坐标
        descripter_rs = descriptor_uv.clone()[0].permute(1, 2, 0)
        descripter_sample = img_utils.get_colors_from_positions(xyc*(cfg.uv_size-1), descripter_rs)
        return descripter_sample



    def projection(self, points, verts, points_inside_mesh_approx=True, scaling_factor=50, revise=True, batch=None, faces_idx=None, verts_uvs=None,faces_t=None, garment=False):
        # STEP 0: preparation
        faces = faces_idx[None, ...].repeat(len(verts), 1, 1)
        meshes = Meshes(verts=verts*scaling_factor, faces=faces)#0.001s
        pcls = Pointclouds(points=points[None]*scaling_factor)
        # compute nearest faces
        # idx:(batch*65536,)，pcls中每个点最近的mesh的face的index
        _, idx = point_mesh_face_distance(meshes, pcls)#0.003s

        triangles_meshes = meshes.verts_packed()[meshes.faces_packed()]  # [batch*13776, 3, 3][面序号, 顶点序号， 坐标轴序号]
        triangles = triangles_meshes[idx]  # [batch*65536, 3, 3]

        # STEP 1: Compute the nearest point on the mesh surface
        # time_before_nearest = time.time()
        nearest, stats = self._parse_nearest_projection(triangles, pcls.points_packed())#0.017s
        # print(f"--time for nearest:{time.time() - time_before_nearest:.4f} seconds")
        if cfg.use_signed_h:
            if garment:
                sign_tensor = self._calculate_points_inside_meshes(pcls.points_packed(), meshes)
            else:
                sign_tensor = self._calculate_points_inside_meshes_normals(pcls.points_packed(), nearest, triangles, meshes.verts_normals_packed()[meshes.faces_packed()][idx])#0.002s
        else:
            sign_tensor = 1.

        # STEP 2-6: Compute the final projection point (check self._revise_nearest() for details)
        dist = torch.norm(pcls.points_packed() - nearest, dim=1) # 采样点距离mesh最近点的距离

        h = dist * sign_tensor

        triangles = triangles_meshes[idx]
        barycentric = points_to_barycentric(triangles, nearest)#0.001s

        # bad case
        barycentric = torch.clamp(barycentric, min=0.)
        barycentric = barycentric / (torch.sum(barycentric, dim=1, keepdim=True) + 1e-12)

        h = h.view(len(verts), -1, 1)
        barycentric = barycentric.view(len(verts), -1, 3)
        idx = idx.view(len(verts), -1)

        # revert scaling
        h = h / scaling_factor
        nearest = nearest / scaling_factor
        # h: [batch, 65536, 1]：采样点到mesh的距离
        # barycentric: [batch, 65536, 3]：采样点在mesh上的三角形的重心坐标
        # idx: [batch, 65536]：采样点最近的mesh的face的index
        # local_coordinates: [batch, 65536, 3, 3]：每个采样点对应的mesh的局部坐标系(normals, tangents, bitangents)
        # nearest_new: [batch, 65536, 3]：采样点在mesh上的投影点（最终选定的插值点）
        # nearest: [batch, 65536, 3]：采样点在mesh上的最近点
        if verts_uvs is None:
            return h, barycentric, idx, nearest
        else:
            # local_coordinates
            local_coordinates_meshes = self._calculate_local_coordinates_meshes(meshes.faces_normals_packed(), triangles_meshes,verts_uvs=verts_uvs,faces_t=faces_t)#0.001s
            if torch.isnan(local_coordinates_meshes).any():
                print('local_coordinates_meshes got nan')
            local_coordinates = local_coordinates_meshes[idx] # [65536,3,3] [normals, tangents, bitangents]
            return h, barycentric, idx, local_coordinates, nearest


    # precise inside/outside computation based on ray-tracing
    def _calculate_points_inside_meshes(self, points, meshes):

        verts_trimesh = meshes.verts_packed().to('cpu').detach().numpy().copy()
        faces_trimesh = meshes.faces_packed().squeeze(0).to('cpu').detach().numpy().copy()
        mesh = trimesh.Trimesh(vertices=verts_trimesh, faces=faces_trimesh, process=False)
        trimesh_intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
        points_trimesh = points.to('cpu').detach().numpy().copy()

        contains = trimesh_intersector.contains_points(points_trimesh)  # [n, ] bool
        contains = torch.tensor(contains, device=points.device)
        contains = 1 - 2*contains  # {-1, 1}, -1 for inside, +1 for outside

        return contains

    def _calculate_points_inside_meshes_kaolin(self, points, meshes):
        pts_signs = 2.0 * (check_sign(meshes.verts_packed()[None], meshes.faces_packed(), points[None]).float() - 0.5)
        return pts_signs[0]

    # approximate inside/outside computation using vertex normals
    def _calculate_points_inside_meshes_normals(self, points, nearest, triangles, normals_triangles):
        barycentric = points_to_barycentric(triangles, nearest)
        normal_at_s = barycentric_to_points(normals_triangles, barycentric)
        contains = ((points - nearest) * normal_at_s).sum(1) < 0.0
        contains = 1 - 2*contains
        return contains

    def _calculate_parallel_triangles(self, points, triangles, verts_normals, faces_normals):

        batch_dim = points.shape[:-1]

        # if batch dim is larger than 1:
        if points.dim() > 2:
            points = points.view(-1, 3)
            triangles = triangles.view(-1, 3, 3)
            verts_normals = verts_normals.view(-1, 3, 3)
            faces_normals = faces_normals.view(-1, 3)

        dist = ((points-triangles[:, 0]) * faces_normals).sum(1)
        verts_normals_cosine = (verts_normals * faces_normals.unsqueeze(1)).sum(2)  # [batch*65536, 3]
        triangles_parallel = triangles + verts_normals * (dist.view(-1, 1) / (verts_normals_cosine + 1e-12)).unsqueeze(2)  # [batch*13776, 3, 3]

        return triangles_parallel.view(*batch_dim, 3, 3)

    def _calculate_points_inside_target_volume(self, points, triangles, verts_normals, faces_normals, return_barycentric=False):

        batch_dim = points.shape[:-1]
        # if batch dim is larger than 1:
        if points.dim() > 2:
            points = points.view(-1, 3)
            triangles = triangles.view(-1, 3, 3)
            verts_normals = verts_normals.view(-1, 3, 3)
            faces_normals = faces_normals.view(-1, 3)

        triangles_parallel = self._calculate_parallel_triangles(
            points, triangles, verts_normals, faces_normals)
        barycentric = points_to_barycentric(triangles_parallel, points)
        inside = torch.prod(barycentric > 0, dim=1)

        if return_barycentric:
            return inside.view(*batch_dim), barycentric.view(*batch_dim, -1)
        else:
            return inside.view(*batch_dim)

    def _align_verts_normals(self, verts_normals, triangles, ps_sign):

        batch_dim = verts_normals.shape[:-2]
        # if batch dim is larger than 1:
        if verts_normals.dim() > 3:
            triangles = triangles.view(-1, 3, 3)
            verts_normals = verts_normals.view(-1, 3, 3)
            ps_sign = ps_sign.unsqueeze(1).repeat(1, batch_dim[1]).view(-1)

        # revert the direction if points inside the mesh
        verts_normals_signed = verts_normals*ps_sign.view(-1, 1, 1)

        edge1 = triangles - triangles[:, [1, 2, 0]]
        edge2 = triangles - triangles[:, [2, 0, 1]]

        # norm edge direction
        edge1_dir = F.normalize(edge1, dim=2)
        edge2_dir = F.normalize(edge2, dim=2)

        # project verts normals onto triangle plane
        faces_normals = torch.cross(triangles[:, 0]-triangles[:, 2], triangles[:, 1]-triangles[:, 0], dim=1)
        verts_normals_projected = verts_normals_signed - torch.sum(verts_normals_signed*faces_normals.unsqueeze(1), dim=2, keepdim=True)*faces_normals.unsqueeze(1)

        p = torch.sum(edge1_dir*verts_normals_projected, dim=2, keepdim=True)
        q = torch.sum(edge2_dir*verts_normals_projected, dim=2, keepdim=True)
        r = torch.sum(edge1_dir*edge2_dir, dim=2, keepdim=True)

        inv_det = 1 / (1 - r**2 + 1e-9)
        c1 = inv_det * (p - r*q)
        c2 = inv_det * (q - r*p)

        # only align inside normals
        c1 = torch.clamp(c1, max=0.)
        c2 = torch.clamp(c2, max=0.)

        verts_normals_aligned = verts_normals_signed - c1*edge1_dir - c2*edge2_dir
        verts_normals_aligned = F.normalize(verts_normals_aligned, eps=1e-12, dim=2)

        # revert the normals direction
        verts_normals_aligned = verts_normals_aligned*ps_sign.view(-1, 1, 1)

        return verts_normals_aligned.view(*batch_dim, 3, 3)

    # Code modified from function closest_point in Trimesh: https://github.com/mikedh/trimesh/blob/main/trimesh/triangles.py#L544
    def _parse_nearest_projection(self, triangles, points, eps=1e-12):
        time_init = time.time()
        # store the location of the closest point
        result = torch.zeros_like(points, device=points.device)
        remain = torch.ones(len(points), dtype=torch.bool, device=points.device)
        # get the three points of each triangle
        # use the same notation as RTCD to avoid confusion
        a = triangles[:, 0, :]
        b = triangles[:, 1, :]
        c = triangles[:, 2, :]
        # check if P is in vertex region outside A
        ab = b - a
        ac = c - a
        ap = points - a
        # this is a faster equivalent of:
        # diagonal_dot(ab, ap)
        d1 = torch.sum(ab * ap, dim=-1)
        d2 = torch.sum(ac * ap, dim=-1)
        # is the point at A #即角pab, pac都是锐角
        # is_a = torch.logical_and(d1 < eps, d2 < eps)
        is_a = (d1 < eps) & (d2 < eps)
        result[is_a] = a[is_a]#将pab,pac都是锐角的a固定住
        remain[is_a] = False
        # check if P in vertex region outside B
        bp = points - b
        d3 = torch.sum(ab * bp, dim=-1)
        d4 = torch.sum(ac * bp, dim=-1)
        # do the logic check
        is_b = (d3 > -eps) & (d4 <= d3) & remain
        result[is_b] = b[is_b]# 固定b
        remain[is_b] = False
        # check if P in edge region of AB, if so return projection of P onto A
        vc = (d1 * d4) - (d3 * d2)
        is_ab = ((vc < eps) &
                 (d1 > -eps) &
                 (d3 < eps) & remain)
        v = (d1[is_ab] / (d1[is_ab] - d3[is_ab])).view((-1, 1))
        result[is_ab] = a[is_ab] + (v * ab[is_ab])
        remain[is_ab] = False
        # check if P in vertex region outside C
        cp = points - c
        d5 = torch.sum(ab * cp, dim=-1)
        d6 = torch.sum(ac * cp, dim=-1)
        is_c = (d6 > -eps) & (d5 <= d6) & remain
        result[is_c] = c[is_c]
        remain[is_c] = False
        # check if P in edge region of AC, if so return projection of P onto AC
        vb = (d5 * d2) - (d1 * d6)
        is_ac = (vb < eps) & (d2 > -eps) & (d6 < eps) & remain
        w = (d2[is_ac] / (d2[is_ac] - d6[is_ac])).view((-1, 1))
        result[is_ac] = a[is_ac] + w * ac[is_ac]
        remain[is_ac] = False
        # check if P in edge region of BC, if so return projection of P onto BC
        va = (d3 * d6) - (d5 * d4)
        is_bc = ((va < eps) &
                 ((d4 - d3) > - eps) &
                 ((d5 - d6) > -eps) & remain)
        d43 = d4[is_bc] - d3[is_bc]
        w = (d43 / (d43 + (d5[is_bc] - d6[is_bc]))).view((-1, 1))
        result[is_bc] = b[is_bc] + w * (c[is_bc] - b[is_bc])
        remain[is_bc] = False
        # any remaining points must be inside face region
        # point is inside face region
        denom = 1.0 / (va[remain] + vb[remain] + vc[remain])
        v = (vb[remain] * denom).reshape((-1, 1))
        w = (vc[remain] * denom).reshape((-1, 1))
        # compute Q through its barycentric coordinates
        result[remain] = a[remain] + (ab[remain] * v) + (ac[remain] * w)
        stats = {
            'is_a': is_a,
            'is_b': is_b,
            'is_c': is_c,
            'is_bc': is_bc,
            'is_ac': is_ac,
            'is_ab': is_ab,
            'remain': remain
        }

        return result, stats

    def _revise_nearest(self,
                        points,
                        idx,
                        meshes,
                        inside,
                        nearest,
                        dist,
                        stats,
                        solve_corner_case=False):

        triangles_meshes = meshes.verts_packed()[meshes.faces_packed()]  # [batch*13776, 3, 3]
        faces_normals_meshes = meshes.faces_normals_packed()
        verts_normals_meshes = meshes.verts_normals_packed()[meshes.faces_packed()]

        # idx, nearest, dist, triangles_meshes, faces_normals_meshes, verts_normals_meshes


        bc_ca_ab = self.faces_to_corres_edges[idx]
        a_b_c = meshes.faces_packed()[idx]

        is_a, is_b, is_c = stats['is_a'], stats['is_b'], stats['is_c']
        is_bc, is_ac, is_ab = stats['is_bc'], stats['is_ac'], stats['is_ab']

        nearest_new, dist_new, idx_new = nearest.clone(), dist.clone(), idx.clone()

        def _revise(is_x, x_idx, x_type):

            points_is_x = points[is_x]
            inside_is_x = inside[is_x]
            if x_type == 'verts':
                verts_is_x = a_b_c[is_x][:, x_idx]
                corres_faces_is_x = self.verts_to_corres_faces[verts_is_x]
                N_repeat = 9  # maximum # of adjacent faces for verts
            elif x_type == 'edges':
                edges_is_x = bc_ca_ab[is_x][:, x_idx]
                corres_faces_is_x = self.edges_to_corres_faces[edges_is_x]
                N_repeat = 2  # maximum # of adjacent faces for edges
            else:
                raise ValueError('x_type should be verts or edges')

            # STEP 2: Find a set T of all triangles containing s~
            triangles_is_x = triangles_meshes[corres_faces_is_x]
            verts_normals_is_x = verts_normals_meshes[corres_faces_is_x]
            faces_normals_is_x = faces_normals_meshes[corres_faces_is_x]

            # STEP 3: Vertex normal alignment
            verts_normals_is_x_aligned = self._align_verts_normals(verts_normals_is_x, triangles_is_x, inside_is_x)

            # STEP 4: Check if inside control volume
            points_is_x_repeated = points_is_x.unsqueeze(1).repeat(1, N_repeat, 1)
            inside_control_volume, barycentric = \
                self._calculate_points_inside_target_volume(points_is_x_repeated, triangles_is_x, verts_normals_is_x_aligned, faces_normals_is_x, return_barycentric=True)  # (n', N_repeat):bool, (n', N_repeat, 3)
            barycentric = torch.clamp(barycentric, min=0.)
            barycentric = barycentric / (torch.sum(barycentric, dim=-1, keepdim=True) + 1e-12)

            # STEP 5: compute set of candidate surface points {s}
            surface_points_set = (barycentric[..., None] * triangles_is_x).sum(dim=2)
            # fixme：这里会产生数值很大的h，不符合实际情况
            # [11765, 9]
            surface_to_points_dist_set = torch.norm(points_is_x_repeated - surface_points_set, dim=2) + 1e10 * (1 - inside_control_volume)  # [n', N_repeat]

            _, idx_is_x = torch.min(surface_to_points_dist_set, dim=1)  # [n', ]

            # STEP 6: Choose the nearest point to x from {s} as the final projection point
            surface_points = surface_points_set[torch.arange(len(idx_is_x)), idx_is_x]  # [n', 3]
            surface_to_points_dist = surface_to_points_dist_set[torch.arange(len(idx_is_x)), idx_is_x]  # [n', ]
            if solve_corner_case:
                corner_case = surface_to_points_dist > 1e9
                surface_to_points_dist[corner_case] = torch.norm(points_is_x_repeated - surface_points_set, dim=2)[torch.arange(len(idx_is_x)), idx_is_x][corner_case]
            faces_is_x = corres_faces_is_x[torch.arange(len(idx_is_x)), idx_is_x]

            # update
            nearest_new[is_x] = surface_points
            dist_new[is_x] = surface_to_points_dist
            idx_new[is_x] = faces_is_x

        # revise verts
        if torch.any(is_a): _revise(is_a, 0, 'verts')
        if torch.any(is_b): _revise(is_b, 1, 'verts')
        if torch.any(is_c): _revise(is_c, 2, 'verts')

        # revise edges
        if torch.any(is_bc): _revise(is_bc, 0, 'edges')
        if torch.any(is_ac): _revise(is_ac, 1, 'edges')
        if torch.any(is_ab): _revise(is_ab, 2, 'edges')

        return nearest_new, dist_new, idx_new

    def _calculate_local_coordinates_meshes(self, faces_normals_meshes, triangles_meshes, faces_idx=None, verts_uvs=None,faces_t=None):
        tangents, bitangents = self._compute_tangent_bitangent(triangles_meshes, faces_idx=faces_idx, verts_uvs=verts_uvs,faces_t=faces_t)
        local_coordinates_meshes = torch.stack((faces_normals_meshes, tangents, bitangents), dim=1)  # [N, 3, 3]
        if torch.isnan(local_coordinates_meshes).any():
            print('local_coordinates_meshes got nan')
        return local_coordinates_meshes

    def _compute_tangent_bitangent(self, triangles_meshes, faces_idx=None, verts_uvs=None,faces_t=None,):
        # Compute tangets and bitangents following:
        # https://learnopengl.com/Advanced-Lighting/Normal-Mapping
        if faces_idx is None:
            face_uv = (verts_uvs[faces_t])  # (13776, 3, 2)
        else:
            face_uv = (verts_uvs[faces_t[faces_idx]])  # (13776, 3, 2)
        face_xyz = triangles_meshes  # (13776, 3, 3)
        assert face_uv.shape[-2:] == (3, 2)
        assert face_xyz.shape[-2:] == (3, 3)
        assert face_uv.shape[:-2] == face_xyz.shape[:-2]
        uv0, uv1, uv2 = face_uv.unbind(-2)
        v0, v1, v2 = face_xyz.unbind(-2)
        duv10 = uv1 - uv0
        duv20 = uv2 - uv0
        duv10x = duv10[..., 0:1]
        duv10y = duv10[..., 1:2]
        duv20x = duv20[..., 0:1]
        duv20y = duv20[..., 1:2]
        det = duv10x * duv20y - duv20x * duv10y
        f = 1.0 / (det + 1e-6)
        dv10 = v1 - v0
        dv20 = v2 - v0
        tangents = f * (duv20y * dv10 - duv10y * dv20)
        bitangents = f * (-duv20x * dv10 + duv10x * dv20)
        tangents = F.normalize(tangents, p=2, dim=-1, eps=1e-6)
        bitangents = F.normalize(bitangents, p=2, dim=-1, eps=1e-6)
        if torch.isnan(tangents).any():
            print('tangents got nan')
        if torch.isnan(bitangents).any():
            print('bitangents got nan')
        return tangents, bitangents

    # parsing mesh (e.g. adjacency of faces, verts, edges, etc.)
    def _parse_mesh(self, verts, faces_idx, N_repeat_edges=2, N_repeat_verts=9):

        meshes = Meshes(verts=[verts], faces=[faces_idx])
        print('parsing mesh topology...')

        # compute faces_to_corres_edges
        faces_to_corres_edges = meshes.faces_packed_to_edges_packed()  # (13776, 3)

        # compute edges_to_corres_faces
        edges_to_corres_faces = torch.full((len(meshes.edges_packed()), N_repeat_edges), -1.0, device=self.device)  # (20664, 2)
        edges_to_corres_verts = torch.full((len(meshes.edges_packed()), 2), -1.0, device=self.device)  # (20664, 2)
        from collections import defaultdict
        for i in range(len(faces_to_corres_edges)):# 遍历每一个face
            for e in faces_to_corres_edges[i]:# 遍历face对应的三个edge序号
                idx = 0
                while idx < edges_to_corres_faces.shape[1]: #收集到足够多的face就退出
                    if edges_to_corres_faces[e][idx] < 0:# 如果当前edge对应的face还没有被添加
                        edges_to_corres_faces[e][idx] = i# 将当前face的index加入到当前edge中
                        if idx == 1:
                            curr_verts = defaultdict(int)
                            for f in edges_to_corres_faces[e]:
                                for v in faces_idx[f.int().item()]:
                                    curr_verts[v.item()] += 1
                            edges_to_corres_verts[e] = torch.tensor([v for v in curr_verts.keys() if curr_verts[v] == 2], device=self.device)
                        break
                    else:
                        idx += 1
                    if idx == edges_to_corres_faces.shape[1]:
                        print('edges_to_corres_faces got too many faces')

        # compute verts_to_corres_faces
        verts_to_corres_faces = torch.full((len(verts), N_repeat_verts), -1.0, device=self.device)  # (6890, 9)
        for i in range(len(faces_idx)):
            for v in faces_idx[i]:
                idx = 0
                while idx < verts_to_corres_faces.shape[1]:
                    if verts_to_corres_faces[v][idx] < 0:
                        verts_to_corres_faces[v][idx] = i
                        break
                    else:
                        idx += 1
                    if idx == verts_to_corres_faces.shape[1]:
                        # print(i,faces_idx[i])
                        print('verts_to_corres_faces got too many faces')

        for i in range(len(faces_idx)):
            for v in faces_idx[i]:
                verts_to_corres_faces[v][verts_to_corres_faces[v] < 0] = verts_to_corres_faces[v][0].clone()

        return faces_to_corres_edges, edges_to_corres_faces, verts_to_corres_faces, edges_to_corres_verts

