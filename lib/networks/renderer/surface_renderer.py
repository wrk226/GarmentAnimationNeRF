import sys
import sys
from lib.config import cfg, args
import numpy
import torch
from pytorch3d.io import load_obj
import os
import numpy as np
import matplotlib.pyplot as plt
import json
# from utils import pytorch3d_rasterize, face_vertices, batch_orth_proj, batch_orth_proj_inv, perspective_project, perspective_project_inv
# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    OrthographicCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
    TexturesVertex,

)

from pytorch3d.datasets import BlenderCamera

import matplotlib.pyplot as plt
import numpy
import torch
from pytorch3d.io import load_obj
import os
import numpy as np
# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes
import time
# Data structures and functions for rendering
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
AmbientLights,
SoftPhongShader
)
from pytorch3d.renderer.mesh import rasterize_meshes
from pytorch3d.renderer.cameras import PerspectiveCameras, OrthographicCameras
from pytorch3d.renderer.mesh.rasterizer import (
    MeshRasterizer,
    RasterizationSettings,
)
import torch.nn.functional as F

from pytorch3d.ops.interp_face_attrs import interpolate_face_attributes
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import Meshes
from pytorch3d.renderer import PointLights, Materials, TexturesUV, SoftPhongShader, MeshRenderer,AmbientLights
import pytorch3d.structures as struct
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import torch
import os
from lib.utils import surface_render_utils

class SurfaceRenderer():
    def __init__(self, annots_path, height=1024, width=1024, light_type='point'):
        self.scale = .5
        height, width = int(height * self.scale), int(width * self.scale)
        if light_type == 'point':
            self.lights = PointLights(device='cuda', location=[[2.5, 2.5, 2.5]],
                                      ambient_color=((0.5, 0.5, 0.5),),

                                        diffuse_color=((0.5, 0.5, 0.5),),
                                        specular_color=((0., 0., 0.),),)
        elif light_type == 'ambient':
            self.lights = AmbientLights(device='cuda', ambient_color=((1.0, 1.0, 1.0),))
        self.raster_settings = RasterizationSettings(image_size=(height, width),
                                            blur_radius=0,
                                            faces_per_pixel=1,
                                            bin_size=None)
        self.height, self.width = height, width
        self.cache = {}
        self.cost_time = 0
        self.annots_path = annots_path
        # self.faces = faces
        annots = np.load(annots_path, allow_pickle=True).item()
        self.cameras = annots['cams']

        # annots_rot = np.load(annots_path.replace('annots','annots_cam_25'), allow_pickle=True).item()
        # cam_rot = annots_rot['cams']
        # for key in ['R', 'T', 'K', 'D']:
        #     self.cameras[key][0] = cam_rot[key][0]

        light_positions = torch.tensor(
                                    #             [
                                    #
                                    # [-2.5, 2.5, -3.],
                                    # [2.5, 2.5, -3.],
                                    # [-2.5, -2.5, -3.],
                                    # [2.5, -2.5, -3.],
                                    # ]
                                    [

                                    [-2.5, 2.5, 2.5],
                                    [2.5, 2.5, 2.5],
                                    [-2.5, -2.5, 2.5],
                                    [2.5, -2.5, 2.5],
                                    ]
            #             [
            # [-5, 5, -5],
            # [5, 5, -5],
            # [-5, -5, -5],
            # [5, -5, -5],
            # [0, 0, -5],
            # ]
                                )[None,:,:]
        light_intensities = torch.ones_like(light_positions).float()*1.7#*2.5#1.7
        self.lights_multi = torch.cat((light_positions, light_intensities), 2).cuda()

    def render(self, vertices, faces, frame_ind, cam_ind, texture=None, extra_rot=None, mode='train', is_garment=False):
        tmp_key = f'{frame_ind.item()}_{cam_ind.item()}'

        if is_garment:
            tmp_key += '_garment'
        # if texture is None:
        #     texture = TexturesVertex(verts_features=torch.ones(1, vertices.shape[1], 3, device='cuda')/2)
        texture = TexturesVertex(verts_features=texture)
        # caculate the transformation from world to image space
        K = self.cameras['K'][cam_ind]*self.scale
        R = self.cameras['R'][cam_ind]
        if mode != 'train' and extra_rot is not None:
            add_degree = -6 * extra_rot.cpu().item()
            R = np.dot(R, cv2.Rodrigues(np.array([0, 0, np.deg2rad(add_degree)]))[0])
        T = self.cameras['T'][cam_ind] / 1000
        pytorch3d_K = self.set_pytorch3d_intrinsic_matrix(K, self.height, self.width)
        cameras = PerspectiveCameras(device='cuda',
                             K=pytorch3d_K[None].astype(np.float32),
                             R=R.T[None].astype(np.float32),
                             T=T.T.astype(np.float32))
        rasterizer = MeshRasterizer(cameras=cameras,
                            raster_settings=self.raster_settings)
        meshes = Meshes(
            verts=vertices,
            faces=faces
        )
        # set the shader
        shader = SoftPhongShader(device='cuda', cameras=cameras, lights=self.lights)

        # 训练模式时将渲染结果缓存
        # if tmp_key in self.cache and mode == 'train':
        #     # print("Use cache!")
        #
        #     fragments = self.cache[tmp_key]
        #     # print('add')
        # else:
        #     fragments = rasterizer(meshes)
        #     # rasterize and shade
        #     self.cache[tmp_key] = fragments
        fragments = rasterizer(meshes)

        meshes.textures = texture
        image = shader(fragments, meshes).squeeze()[..., :3]
        depth = fragments.zbuf.squeeze()
        return image, depth

    def render_with_texture(self, vertices, faces, frame_ind, cam_ind, texture=None, extra_rot=None, mode='train', is_garment=False, shift=False):
        # Attributes
        face_vertices = surface_render_utils.face_vertices(vertices, faces)
        normals = surface_render_utils.vertex_normals(vertices,faces); face_normals = surface_render_utils.face_vertices(normals, faces)
        transformed_vertices = vertices.clone()
        if shift:
            transformed_vertices[:,:,2] = transformed_vertices[:,:,2] - transformed_vertices[:,:,2].min()
            transformed_vertices[:,:,2] = transformed_vertices[:,:,2]/transformed_vertices[:,:,2].max()*80 + 10
        transformed_normals = surface_render_utils.vertex_normals(transformed_vertices, faces); transformed_face_normals = surface_render_utils.face_vertices(transformed_normals, faces)

        attributes = surface_render_utils.face_vertices(texture, faces)

        # caculate the transformation from world to image space
        K = self.cameras['K'][cam_ind]*self.scale
        R = self.cameras['R'][cam_ind]
        if mode != 'train' and extra_rot is not None:
            add_degree = -6 * extra_rot.cpu().item()
            R = np.dot(R, cv2.Rodrigues(np.array([0, 0, np.deg2rad(add_degree)]))[0])
        T = self.cameras['T'][cam_ind] / 1000
        pytorch3d_K = self.set_pytorch3d_intrinsic_matrix(K, self.height, self.width)
        cameras = PerspectiveCameras(device='cuda',
                             K=pytorch3d_K[None].astype(np.float32),
                             R=R.T[None].astype(np.float32),
                             T=T.T.astype(np.float32))

        # transform vertices according to camera
        verts_view = cameras.get_world_to_view_transform().transform_points(transformed_vertices)
        to_ndc_transform = cameras.get_ndc_camera_transform()
        verts_proj = cameras.transform_points(transformed_vertices)
        verts_ndc = to_ndc_transform.transform_points(verts_proj)
        verts_ndc[..., 2] = verts_view[..., 2]

        # rasterize
        image = self.pytorch3d_rasterize(verts_ndc, faces, image_size=self.height, attributes=attributes)
        alpha_image = image[:,[-1]]
        image = image[:,:3]*alpha_image + torch.ones_like(alpha_image)*(1-alpha_image)

        return image

    def render_with_light(self, vertices, faces, frame_ind, cam_ind, texture=None, extra_rot=None, mode='train', is_garment=False, shift=False, return_normal=False,
                          height=-1, width=-1):
        if height == -1:
            height = self.height
        if width == -1:
            width = self.width
        self.scale = height / 1024
        # TODO: texture is not used here, because our body don't have texture
        # Attributes
        face_vertices = surface_render_utils.face_vertices(vertices, faces)
        normals = surface_render_utils.vertex_normals(vertices,faces)
        face_normals = surface_render_utils.face_vertices(normals, faces)
        transformed_vertices = vertices.clone()
        if shift:
            transformed_vertices[:,:,2] = transformed_vertices[:,:,2] - transformed_vertices[:,:,2].min()
            transformed_vertices[:,:,2] = transformed_vertices[:,:,2]/transformed_vertices[:,:,2].max()*80 + 10
        transformed_normals = surface_render_utils.vertex_normals(transformed_vertices, faces); transformed_face_normals = surface_render_utils.face_vertices(transformed_normals, faces)
        face_colors = torch.ones_like(face_vertices)*180/255.


        attributes = torch.cat([face_colors,
                        transformed_face_normals.detach(),
                        face_vertices.detach(),
                        face_normals],
                        -1)

        # caculate the transformation from world to image space
        # todo:
        # K = self.cameras['K'][cam_ind]*self.scale
        # IndexError: list index out of range
        K = np.array(self.cameras['K'][cam_ind])*self.scale
        # except:
        #     print(cam_ind, len(self.cameras['K']))
        R = np.array(self.cameras['R'][cam_ind])

        x_rot,y_rot,z_rot = cfg.novel_pose_rotation
        R = np.dot(R, cv2.Rodrigues(np.array([np.deg2rad(x_rot), np.deg2rad(y_rot), np.deg2rad(z_rot)]))[0])

        if mode != 'train' and extra_rot is not None:
            add_degree = -6 * extra_rot.cpu().item()
            R = np.dot(R, cv2.Rodrigues(np.array([0, 0, np.deg2rad(add_degree)]))[0])

        T = np.array(self.cameras['T'][cam_ind]) / 1000
        pytorch3d_K = self.set_pytorch3d_intrinsic_matrix(K, height, width)
        cameras = PerspectiveCameras(device='cuda',
                             K=pytorch3d_K[None].astype(np.float32),
                             R=R.T[None].astype(np.float32),
                             T=T.T.astype(np.float32))

        # transform vertices according to camera
        verts_view = cameras.get_world_to_view_transform().transform_points(transformed_vertices)
        to_ndc_transform = cameras.get_ndc_camera_transform()
        verts_proj = cameras.transform_points(transformed_vertices)
        verts_ndc = to_ndc_transform.transform_points(verts_proj)
        verts_ndc[..., 2] = verts_view[..., 2]

        # rasterize
        rendering, ret = self.pytorch3d_rasterize(verts_ndc, faces, image_size=height, attributes=attributes)
        ####
        alpha_images = rendering[:, -2, :, :][:, None, :, :].detach()

        # albedo
        albedo_images = rendering[:, :3, :, :]
        # plt.imshow(albedo_images.cpu().numpy()[0].transpose(1,2,0))
        # plt.show()
        # mask
        transformed_normal_map = rendering[:, 3:6, :, :].detach()
        pos_mask = (transformed_normal_map[:, 2:, :, :] < 0.15).float()

        # shading
        normal_images = rendering[:, 9:12, :, :].detach()
        vertice_images = rendering[:, 6:9, :, :].detach()

        shading = self.add_directionlight(normal_images.permute(0,2,3,1).reshape([1, -1, 3]), self.lights_multi)
        shading_images = shading.reshape([1, albedo_images.shape[2], albedo_images.shape[3], 3]).permute(0,3,1,2).contiguous()
        shaded_images = albedo_images*shading_images

        # Gamma correction
        gamma = 2.2
        shaded_images = shaded_images ** (1.0 / gamma)

        # black background
        shape_images = shaded_images*alpha_images + torch.zeros_like(shaded_images).to(vertices.device)*(1-alpha_images)
        # white background
        # shape_images = shaded_images*alpha_images + torch.ones_like(shaded_images).to(vertices.device)*(1-alpha_images)
        shape_images = shape_images.squeeze().permute(1,2,0)

        # depth
        depth_images = rendering[:, -1, :, :].squeeze()
        if return_normal:
            return shape_images, depth_images, normal_images.squeeze().permute(1,2,0)

        return shape_images, depth_images, ret
        # return shape_images, depth_images

    def render_with_light_and_texture(self, vertices, faces, frame_ind, cam_ind, texture=None, extra_rot=None, mode='train', is_garment=False, shift=False, return_normal=False,
                      height=-1, width=-1):
        if height == -1:
            height = self.height
        if width == -1:
            width = self.width
        self.scale = height / 1024
        # TODO: texture is not used here, because our body don't have texture
        face_colors = surface_render_utils.face_vertices(texture, faces)
        # Attributes
        face_vertices = surface_render_utils.face_vertices(vertices, faces)
        normals = surface_render_utils.vertex_normals(vertices,faces)
        face_normals = surface_render_utils.face_vertices(normals, faces)
        transformed_vertices = vertices.clone()
        if shift:
            transformed_vertices[:,:,2] = transformed_vertices[:,:,2] - transformed_vertices[:,:,2].min()
            transformed_vertices[:,:,2] = transformed_vertices[:,:,2]/transformed_vertices[:,:,2].max()*80 + 10
        transformed_normals = surface_render_utils.vertex_normals(transformed_vertices, faces); transformed_face_normals = surface_render_utils.face_vertices(transformed_normals, faces)
        # face_colors = torch.ones_like(face_vertices)*180/255.


        attributes = torch.cat([face_colors,
                        transformed_face_normals.detach(),
                        face_vertices.detach(),
                        face_normals],
                        -1)

        # caculate the transformation from world to image space
        # todo:
        # K = self.cameras['K'][cam_ind]*self.scale
        # IndexError: list index out of range
        K = np.array(self.cameras['K'][cam_ind])*self.scale
        # except:
        #     print(cam_ind, len(self.cameras['K']))
        R = np.array(self.cameras['R'][cam_ind])
        if mode != 'train' and extra_rot is not None:
            add_degree = -6 * extra_rot.cpu().item()
            R = np.dot(R, cv2.Rodrigues(np.array([0, 0, np.deg2rad(add_degree)]))[0])
        T = np.array(self.cameras['T'][cam_ind]) / 1000
        pytorch3d_K = self.set_pytorch3d_intrinsic_matrix(K, height, width)
        cameras = PerspectiveCameras(device='cuda',
                             K=pytorch3d_K[None].astype(np.float32),
                             R=R.T[None].astype(np.float32),
                             T=T.T.astype(np.float32))

        # transform vertices according to camera
        verts_view = cameras.get_world_to_view_transform().transform_points(transformed_vertices)
        to_ndc_transform = cameras.get_ndc_camera_transform()
        verts_proj = cameras.transform_points(transformed_vertices)
        verts_ndc = to_ndc_transform.transform_points(verts_proj)
        verts_ndc[..., 2] = verts_view[..., 2]

        # rasterize
        rendering, ret = self.pytorch3d_rasterize(verts_ndc, faces, image_size=height, attributes=attributes)
        ####
        alpha_images = rendering[:, -2, :, :][:, None, :, :].detach()

        # albedo
        albedo_images = rendering[:, :3, :, :]
        # plt.imshow(albedo_images.cpu().numpy()[0].transpose(1,2,0))
        # plt.show()
        # mask
        transformed_normal_map = rendering[:, 3:6, :, :].detach()
        pos_mask = (transformed_normal_map[:, 2:, :, :] < 0.15).float()

        # shading
        normal_images = rendering[:, 9:12, :, :].detach()
        vertice_images = rendering[:, 6:9, :, :].detach()

        self.lights = AmbientLights(device='cuda', ambient_color=((1.0, 1.0, 1.0),))
        shading = self.add_directionlight(normal_images.permute(0,2,3,1).reshape([1, -1, 3]), self.lights_multi)
        shading_images = shading.reshape([1, albedo_images.shape[2], albedo_images.shape[3], 3]).permute(0,3,1,2).contiguous()
        shaded_images = albedo_images*shading_images

        # Gamma correction
        gamma = 2.2
        shaded_images = shaded_images ** (1.0 / gamma)

        # black background
        shape_images = shaded_images*alpha_images + torch.zeros_like(shaded_images).to(vertices.device)*(1-alpha_images)
        # white background
        # shape_images = shaded_images*alpha_images + torch.ones_like(shaded_images).to(vertices.device)*(1-alpha_images)
        shape_images = shape_images.squeeze().permute(1,2,0)

        # depth
        depth_images = rendering[:, -1, :, :].squeeze()
        if return_normal:
            return shape_images, depth_images, normal_images.squeeze().permute(1,2,0)

        return shape_images, depth_images, ret
        # return shape_images, depth_images

    def pytorch3d_rasterize(self, vertices, faces, image_size, attributes=None,
                        soft=False, blur_radius=0.0, sigma=1e-8, faces_per_pixel=1, gamma=1e-4,
                        perspective_correct=False, clip_barycentric_coords=True):
        fixed_vertices = vertices.clone()
        # fixed_vertices[...,:2] = -fixed_vertices[...,:2]


        meshes_screen = Meshes(verts=fixed_vertices.float(), faces=faces.long())
        # import ipdb; ipdb.set_trace()
        # pytorch3d rasterize
        pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
            meshes_screen,
            image_size=image_size,
            blur_radius=blur_radius,
            faces_per_pixel=faces_per_pixel,
            perspective_correct=perspective_correct,
            clip_barycentric_coords=clip_barycentric_coords,
            # max_faces_per_bin = faces.shape[1],
            bin_size = 0
        )
        # import ipdb; ipdb.set_trace()
        vismask = (pix_to_face > -1).float().squeeze(-1)
        depth = zbuf.squeeze(-1)

        if attributes is None:
            return depth, vismask
        else:
            vismask = (pix_to_face > -1).float()
            D = attributes.shape[-1]
            attributes = attributes.clone()
            attributes = attributes.view(attributes.shape[0]*attributes.shape[1], 3, attributes.shape[-1])
            N, H, W, K, _ = bary_coords.shape
            mask = pix_to_face == -1
            pix_to_face = pix_to_face.clone()
            pix_to_face[mask] = 0
            idx = pix_to_face.view(N * H * W * K, 1, 1).expand(N * H * W * K, 3, D)
            pixel_face_vals = attributes.gather(0, idx).view(N, H, W, K, 3, D)
            pixel_vals = (bary_coords[..., None] * pixel_face_vals).sum(dim=-2)
            pixel_vals[mask] = 0  # Replace masked values in output.
            pixel_vals = pixel_vals[:,:,:,0].permute(0,3,1,2)
            pixel_vals = torch.cat([pixel_vals, vismask[:,:,:,0][:,None,:,:], depth[:, None,:,:]], dim=1)
            ret = {'pix_to_face': pix_to_face.squeeze(),
                   'bary_coords': bary_coords.squeeze(),}
            return pixel_vals, ret

    def add_directionlight(self, normals, lights=None, auto_adjust_light=True):
        '''
            normals: [bz, nv, 3]
            lights: [bz, nlight, 6]
        returns:
            shading: [bz, nv, 3]
        '''
        light_direction = lights[:,:,:3]
        light_intensities = lights[:,:,3:]
        directions_to_lights = F.normalize(light_direction[:,:,None,:].expand(-1,-1,normals.shape[1],-1), dim=3)
        # normals_dot_lights = torch.clamp((normals[:,None,:,:]*directions_to_lights).sum(dim=3), 0., 1.)
        # normals_dot_lights = (normals[:,None,:,:]*directions_to_lights).sum(dim=3)
        normals_dot_lights = torch.clamp((normals[:,None,:,:]*directions_to_lights).sum(dim=3), 0., 1.)
        shading = normals_dot_lights[:,:,:,None]*light_intensities[:,:,None,:]
        return shading.mean(1)

    # def get_msic(self, H, W):
    #     assert H == W
    #     pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
    #         self.mesh,
    #         image_size=H,
    #         blur_radius=0.,#1e-8
    #         faces_per_pixel=1,#50
    #         perspective_correct=False,
    #         clip_barycentric_coords=True,
    #         # max_faces_per_bin = faces.shape[1],
    #         bin_size = 0
    #     )
    #     vismask = (pix_to_face > -1).float().squeeze(-1)
    #     depth = zbuf.squeeze(-1)# 依据mesh算出来的深度
    #     # print(vismask.shape,depth.shape)#torch.Size([1, 400, 400]) torch.Size([1, 400, 400])
    #     # sys.exit()
    #     fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
    #     axes[0].imshow(vismask[0].cpu().numpy())
    #     axes[1].imshow(depth[0].cpu().numpy())
    #     plt.show()
    #     return vismask, depth


    def set_pytorch3d_intrinsic_matrix(self, K, H, W):
        fx = -K[0, 0] * 2.0 / W
        fy = -K[1, 1] * 2.0 / H
        px = -(K[0, 2] - W / 2.0) * 2.0 / W
        py = -(K[1, 2] - H / 2.0) * 2.0 / H
        K = [
            [fx, 0, px, 0],
            [0, fy, py, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ]
        K = np.array(K)
        return K

    def get_camera(self, cam_id, height, width):
        K = np.array(self.cameras['K'][cam_id])*self.scale
        # except:
        #     print(cam_ind, len(self.cameras['K']))
        R = np.array(self.cameras['R'][cam_id])
        T = np.array(self.cameras['T'][cam_id]) / 1000
        pytorch3d_K = self.set_pytorch3d_intrinsic_matrix(K, height, width)
        cameras = PerspectiveCameras(device='cuda',
                             K=pytorch3d_K[None].astype(np.float32),
                             R=R.T[None].astype(np.float32),
                             T=T.T.astype(np.float32))
        return cameras

    def get_visiable_faces(self, verts, faces, cam_id):

        camera = self.get_camera(cam_id, self.height, self.width)
        rasterizer = MeshRasterizer(cameras=camera,
                            raster_settings=self.raster_settings)
        meshes = Meshes(
            verts=verts,
            faces=faces
        )
        fragments = rasterizer(meshes)
        visiable_faces = fragments.pix_to_face.squeeze()
        return visiable_faces

    def get_mesh_normal(self, verts, faces):
        verts_normal = surface_render_utils.vertex_normals(verts, faces)
        return verts_normal

    def rotate_normal_map(self, normal_map, source_cam_id, target_cam_id):
        source_cam = self.get_camera(source_cam_id, self.height, self.width)
        target_cam = self.get_camera(target_cam_id, self.height, self.width)
        source_to_target = source_cam.get_world_to_view_transform().inverse().get_matrix()@target_cam.get_world_to_view_transform().get_matrix()
        source_to_target = source_to_target[0,:3,:3]
        source_to_target = source_to_target

        normal_map = normal_map[..., [0, 2, 1]]
        # normal_map[..., 1] *= -1


        rotated_normal_map = normal_map.reshape(-1,3) @ source_to_target#.transpose(0,1)

        # rotated_normal_map = rotated_normal_map[..., [0, 2, 1]]
        # rotated_normal_map[..., 1] *= -1
        return rotated_normal_map.reshape(1,512,512,3)


# main
if __name__ == '__main__':
    data_root = 'C:/Users/D-Blue/Desktop/cloth/local/surface-aligned-nerf-neus2/data/synthetic_human/garment-body_view20'
    annots_path = os.path.join(data_root, 'annots.npy')
    obj_path = os.path.join(data_root,'body/b_000022.obj')
    verts, faces, aux = load_obj(obj_path)
    verts = verts[:, [0, 2, 1]]
    verts[:, 1] *= -1

    faces = faces.verts_idx.cuda()[None]
    verts = verts.cuda()[None]

    renderer = SurfaceRenderer(annots_path=annots_path)

    visiable_faces = renderer.get_visiable_faces(verts, faces, 0)
    face_normal = surface_render_utils.face_vertices(surface_render_utils.vertex_normals(verts, faces), faces).squeeze()
    face_normal_avg = face_normal.mean(1)
    vis_norm = torch.zeros_like(visiable_faces)[...,None].repeat(1,1,3).float()
    vis_norm[visiable_faces!=-1] = face_normal_avg[visiable_faces[visiable_faces!=-1]]


    rotated_normal_map = renderer.rotate_normal_map(vis_norm[None], 1,0)
    plt.subplot(2,2,1)
    plt.imshow(vis_norm.cpu().numpy())
    plt.subplot(2,2,2)
    plt.imshow(rotated_normal_map[0].cpu().numpy())
    plt.show()
    print(1)





