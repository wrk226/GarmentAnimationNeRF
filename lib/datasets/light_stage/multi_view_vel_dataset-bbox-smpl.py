import matplotlib.pyplot as plt
import torch.utils.data as data
from lib.utils import base_utils
from PIL import Image
import numpy as np
import json
import os
import imageio
import cv2
from lib.config import cfg
from lib.utils.if_nerf import if_nerf_data_utils
import torch
import math
import pickle
from lib.smplmodel.body_model import SMPL, SMPLLayer

class Dataset(data.Dataset):
    def __init__(self, data_root, human, ann_file, split):
        super(Dataset, self).__init__()
        # super().__init__()
        self.img_src = cfg.img_src
        self.data_root = data_root
        self.human = human
        self.split = split

        annots = np.load(ann_file, allow_pickle=True).item()
        self.cams = annots['cams']



        # annots_rot = np.load(ann_file.replace('annots','annots_cam_-25'), allow_pickle=True).item()
        # cam_rot = annots_rot['cams']
        # for key in ['R', 'T', 'K', 'D']:
        #     self.cams[key][1] = cam_rot[key][0]





        num_cams = len(self.cams['K'])
        # test_view = [i for i in range(num_cams) if i in cfg.training_view]
        test_view = [i for i in range(num_cams) if i not in cfg.training_view]
        if len(test_view) == 0:
            test_view = [0]
        view = cfg.training_view if split == 'train' else test_view
        # view = cfg.training_view


        # prepare input images
        #　加载前置动作帧
        i = cfg.begin_ith_frame - cfg.prev_frames
        i_intv = cfg.frame_interval
        ni = cfg.num_train_frame + cfg.prev_frames
        if cfg.test_novel_pose:
            i = cfg.begin_ith_frame + cfg.num_train_frame * i_intv
            ni = cfg.num_novel_pose_frame + cfg.prev_frames
            if self.human == 'CoreView_390':
                i = 0

        # 加载reference views
        reference_view = np.array(cfg.reference_view)

        self.reference_ims = []
        for reference_idx in range(len(reference_view)):
            self.reference_ims.append(np.array([
                np.array(ims_data['ims'])[reference_view[[reference_idx for _ in view]]] # 按照视角和指定帧选择对应的图像
                for ims_data in annots['ims'][i:i + ni * i_intv][::i_intv]
            ]).ravel())
        self.reference_cams = []
        for reference_idx in reference_view:
            self.reference_cams.append({'R':annots['cams']['R'][reference_idx],
                                        'T':annots['cams']['T'][reference_idx]/1000.,
                                        'K':annots['cams']['K'][reference_idx],
                                        'D':annots['cams']['D'][reference_idx]})

        # self.reference_front_ims = np.array([
        #     np.array(ims_data['ims'])[reference_view[[0 for _ in view]]] # 按照视角和指定帧选择对应的图像
        #     for ims_data in annots['ims'][i:i + ni * i_intv][::i_intv]
        # ]).ravel()
        # self.reference_back_ims = np.array([
        #     np.array(ims_data['ims'])[reference_view[[1 for _ in view]]] # 按照视角和指定帧选择对应的图像
        #     for ims_data in annots['ims'][i:i + ni * i_intv][::i_intv]
        # ]).ravel()
        # if cfg.n_reference_view == 4:
        #     self.reference_right_ims = np.array([
        #         np.array(ims_data['ims'])[reference_view[[2 for _ in view]]] # 按照视角和指定帧选择对应的图像
        #         for ims_data in annots['ims'][i:i + ni * i_intv][::i_intv]
        #     ]).ravel()
        #     self.reference_left_ims = np.array([
        #         np.array(ims_data['ims'])[reference_view[[3 for _ in view]]] # 按照视角和指定帧选择对应的图像
        #         for ims_data in annots['ims'][i:i + ni * i_intv][::i_intv]
        #     ]).ravel()



        self.ims = np.array([
            np.array(ims_data['ims'])[view] # 按照视角和指定帧选择对应的图像
            for ims_data in annots['ims'][i:i + ni * i_intv][::i_intv]
        ]).ravel()


        # self.ims_back = np.array([
        #     np.array(ims_data['ims'])[(np.array(view)+10)%20] # 按照视角和指定帧选择对应的图像
        #     for ims_data in annots['ims'][i:i + ni * i_intv][::i_intv]
        # ]).ravel()


        self.cam_inds = np.array([
            np.arange(len(ims_data['ims']))[view] # 按照视角和指定帧选择对应的摄像机
            for ims_data in annots['ims'][i:i + ni * i_intv][::i_intv]
        ]).ravel()

        self.num_cams = len(view)
        self.nrays = cfg.N_rand

        # if 'garment' in cfg.train_dataset.data_root:
        #     obj_path = os.path.join(self.data_root, "body", f'b_000001.obj')
        # elif 'skirt' in cfg.train_dataset.data_root:
        #     obj_path = os.path.join(self.data_root, "body", f'b_0000001.obj')
        # elif 'tango' in cfg.train_dataset.data_root:
        #     obj_path = os.path.join(self.data_root, "body", f'b_000001.obj')
        # _, self.faces = self.load_obj(obj_path)

        min_scale = math.sqrt(cfg.N_rand)/512.
        self.grid_sampler = if_nerf_data_utils.FlexGridRaySampler(N_samples=cfg.N_rand,
                                                                 min_scale=min_scale,
                                                                 max_scale=1.,
                                                                 scale_anneal=0.01)#0.0025 )

        # smpl
        self.body_model = SMPLLayer(model_path='smpl/basicmodel_m_lbs_10_207_0_v1.1.0.pkl')




    def load_bigpose(self):
        big_poses = np.zeros([len(self.joints), 3]).astype(np.float32).ravel()
        angle = 30
        big_poses[5] = np.deg2rad(angle)
        big_poses[8] = np.deg2rad(-angle)
        big_poses = big_poses.reshape(-1, 3)
        big_A = if_nerf_data_utils.get_rigid_transformation(
            big_poses, self.joints, self.parents)
        big_A = big_A.astype(np.float32)
        return big_A

    def get_mask(self, name):
        if 'zju' in self.data_root:
            msk_path = os.path.join(self.data_root, 'mask_cihp', name)[:-4] + '.png'
        elif cfg.use_smpl:
            msk_path = os.path.join(self.data_root, 'mask_cihp', name)[:-4] + '.png'
        elif cfg.use_seg_msk:
            msk_path = os.path.join(self.data_root, name.replace("images",'segment'))[:-4] + '.png'

        # else:
        #     msk_path = os.path.join(self.data_root, self.ims[index].replace("images",'mask_cihp'))[:-4] + '.png'

        msk_cihp = imageio.imread(msk_path.replace("\\","/"))

        ##############
        # mask里只有人体的分割，没有不同语意部位的
        # msk_path_old = os.path.join(self.data_root, 'mask', self.ims[index])[:-4] + '.png'
        # msk_cihp_old = imageio.imread(msk_path_old)
        ##############

        if len(msk_cihp.shape) == 3:
            msk_cihp = msk_cihp[..., 0]
        if 'deepcap' in self.data_root or 'nerfcap' in self.data_root:
            msk_cihp = (msk_cihp > 125).astype(np.uint8)
        elif cfg.use_seg_msk:
            pass
        else:
            msk_cihp = (msk_cihp != 0).astype(np.uint8)
        msk = msk_cihp
        orig_msk = msk.copy()

        if cfg.erode_edge:# and not cfg.eval
            border = 5
            kernel = np.ones((border, border), np.uint8)
            msk_erode = cv2.erode(msk.copy(), kernel)
            msk_dilate = cv2.dilate(msk.copy(), kernel)
            msk[(msk_dilate - msk_erode) == 1] = 100

        return msk, orig_msk

    def get_normal(self, index):
        if 'zju' in self.data_root:
            normal_path = os.path.join(self.data_root, 'normals', self.ims[index].replace("\\","/"))[:-4] + '.png'
        elif cfg.use_smpl:
            normal_path = os.path.join(self.data_root, 'normals', self.ims[index].replace("\\","/"))[:-4] + '.png'
        else:
            normal_path = os.path.join(self.data_root, self.ims[index].replace("images",'normal').replace("\\","/"))[:-4] + '.png'
        normal = imageio.imread(normal_path)
        normal = normal.astype(np.float32) / 255. * 2 - 1
        return normal
        # if cfg.use_smpl:
        #     return normal
        # else:
        #     return normal[...,[2,1,0]]

    def load_obj(self, obj_path):
        with open(obj_path) as f:
            lines = f.readlines()
        verts = []
        faces = []
        for line in lines:
            if line.startswith('v '):
                verts.append([float(v) for v in line.split()[1:4]])
            elif line.startswith('f '):
                faces.append([int(v.split('/')[0]) - 1 for v in line.split()[1:4]])
        verts = np.array(verts, dtype=np.float32)
        faces = np.array(faces, dtype=np.int32)
        return verts, faces

    def load_trans(self, offset_path):
        with open(offset_path, 'r') as file:
            offset = json.load(file)
        trans = np.array([offset['offset_x'], offset['offset_y'], 0.]).astype(np.float32)
        return trans

    def prepare_input(self, i):
        if 'walk' in cfg.train_dataset.data_root or 'samba' in cfg.train_dataset.data_root or 'bad' in cfg.train_dataset.data_root:
            obj_path = os.path.join(self.data_root, f"body{cfg.body_suffix}", f'b_{i+1:06d}.obj')
            offset_path = os.path.join(self.data_root, f"offset{cfg.body_suffix}", f'b_{i+1:06d}.json')
            params_path = os.path.join(self.data_root, cfg.param_path+cfg.body_suffix, f'b_{i+1:06d}.pkl')
        elif 'skirt' in cfg.train_dataset.data_root:
            obj_path = os.path.join(self.data_root, f"body{cfg.body_suffix}", f'b_{i:07d}.obj')
            offset_path = os.path.join(self.data_root, f"offset{cfg.body_suffix}", f'b_{i:07d}.json')
            params_path = os.path.join(self.data_root, cfg.param_path+cfg.body_suffix, f'b_{i:07d}.pkl')
        else:
            obj_path = os.path.join(self.data_root, f"body{cfg.body_suffix}", f'b_{i+1:06d}.obj')
            offset_path = os.path.join(self.data_root, f"offset{cfg.body_suffix}", f'b_{i+1:06d}.json')
            params_path = os.path.join(self.data_root, cfg.param_path+cfg.body_suffix, f'b_{i+1:06d}.pkl')
        if os.path.exists(os.path.join(self.data_root, f"body{cfg.body_suffix}_cache", f'v_{i}.npy')):
            overts = np.load(os.path.join(self.data_root, f"body{cfg.body_suffix}_cache", f'v_{i}.npy'))
            faces = np.load(os.path.join(self.data_root, f"body{cfg.body_suffix}_cache", f'f_{i}.npy'))
        else:
            overts, faces = self.load_obj(obj_path)
            os.makedirs(os.path.join(self.data_root, f"body{cfg.body_suffix}_cache"), exist_ok=True)
            np.save(os.path.join(self.data_root, f"body{cfg.body_suffix}_cache", f'v_{i}.npy'), overts)
            np.save(os.path.join(self.data_root, f"body{cfg.body_suffix}_cache", f'f_{i}.npy'), faces)
        # coords system switch: from( x: right, y: up, z: front) to (x: right, y: back, z: up)
        # o: original space
        # w: world space, without offset adjust, may same as scene space
        # s: scene space(camera space, same as image projection space, usually with offset adjust)
        # p: posed space(smpl space, offset adjust + rotation adjust)
        wverts = overts[:, [0, 2, 1]]
        wverts[:, 1] *= -1
        wtrans = self.load_trans(offset_path)
        otrans = wtrans[[0, 2, 1]]
        otrans[2] *= -1
        sverts = wverts + wtrans.reshape(1,3)

        body_info = {
                        'overts':overts,
                        'faces':faces,
                        'wverts':wverts,
                        'wtrans':wtrans,
                        'sverts':sverts
                    }
        with open(params_path, 'rb') as smpl_file:
            params = pickle.load(smpl_file, encoding='latin1')


        # params = np.load(params_path,allow_pickle=True).item()
        # v_cano = self.body_model.inverse_lbs(overts, params)
        ojoints = params['joints'][0,:24].detach().cpu().numpy()
        # load pose
        ptrans = params['joints'][0,0].detach().cpu().numpy()
        prot = params['global_orient'][0,0].detach().cpu().numpy()
        wjoints = ojoints[:,[0,2,1]]
        wjoints[:,1] *= -1
        sjoints = wjoints + wtrans.reshape(1,3)

        body_info.update({
                            'wjoints':wjoints,
                            'sjoints':sjoints})
        # from original vert space to smpl space

        pverts = (overts - ptrans) @ prot
        pjoints = (ojoints - ptrans) @ prot
        body_info.update({
            'pverts': pverts,
            'pjoints': pjoints,
            'prot': prot,
            'ptrans': ptrans
        })
        return body_info, params

    def prepare_coarse_input(self, i):
        if 'walk' in cfg.train_dataset.data_root or 'samba' in cfg.train_dataset.data_root or 'bad' in cfg.train_dataset.data_root:
            obj_path = os.path.join(self.data_root, f"{cfg.garment_template}", f'g_{i:07d}.obj')
            offset_path = os.path.join(self.data_root, "offset", f'b_{i+1:06d}.json')
            params_path = os.path.join(self.data_root, cfg.param_path, f'b_{i+1:06d}.pkl')
        elif 'skirt' in cfg.train_dataset.data_root:
            obj_path = os.path.join(self.data_root, f"body", f'b_{i:07d}.obj')
            offset_path = os.path.join(self.data_root, "offset", f'b_{i:07d}.json')
            params_path = os.path.join(self.data_root, cfg.param_path, f'b_{i:07d}.pkl')

        if os.path.exists(os.path.join(self.data_root, "coarse_garment_cache", f'v_{i}.npy')):
            overts = np.load(os.path.join(self.data_root, "coarse_garment_cache", f'v_{i}.npy'), allow_pickle=True)
            faces = np.load(os.path.join(self.data_root, "coarse_garment_cache", f'f_{i}.npy'), allow_pickle=True)
        else:
            overts, faces = self.load_obj(obj_path)
            os.makedirs(os.path.join(self.data_root, "coarse_garment_cache"), exist_ok=True)
            np.save(os.path.join(self.data_root, "coarse_garment_cache", f'v_{i}.npy'), overts)
            np.save(os.path.join(self.data_root, "coarse_garment_cache", f'f_{i}.npy'), faces)
        # coords system switch: from( x: right, y: up, z: front) to (x: right, y: back, z: up)
        wverts = overts[:, [0, 2, 1]]
        wverts[:, 1] *= -1
        wtrans = self.load_trans(offset_path)
        otrans = wtrans[[0, 2, 1]]
        otrans[2] *= -1
        sverts = wverts + wtrans.reshape(1,3)

        coarse_info = {
                        'faces':faces,
                        'wverts':wverts,
                        'wtrans':wtrans,
                        'sverts':sverts
                    }
        params = np.load(params_path,allow_pickle=True).item()
        # load pose
        ptrans = params['joints'][0]
        prot = params['global_orient']
        # from original vert space to smpl space
        pverts = (overts - ptrans) @ prot
        coarse_info.update({
            'pverts': pverts,
            'prot': prot,
            'ptrans': ptrans
        })
        return coarse_info

    def prepare_sr_input(self, i):
        if 'walk' in cfg.train_dataset.data_root or 'samba' in cfg.train_dataset.data_root or 'bad' in cfg.train_dataset.data_root:
            obj_path = os.path.join(self.data_root, f"body{cfg.body_suffix}", f'b_{i+1:06d}.obj')
            offset_path = os.path.join(self.data_root, f"offset{cfg.body_suffix}", f'b_{i+1:06d}.json')
        elif 'skirt' in cfg.train_dataset.data_root:
            obj_path = os.path.join(self.data_root, f"body{cfg.body_suffix}", f'b_{i:07d}.obj')
            offset_path = os.path.join(self.data_root, f"offset{cfg.body_suffix}", f'b_{i:07d}.json')
        else:
            obj_path = os.path.join(self.data_root, f"body{cfg.body_suffix}", f'b_{i+1:06d}.obj')
            offset_path = os.path.join(self.data_root, f"offset{cfg.body_suffix}", f'b_{i+1:06d}.json')


        if os.path.exists(os.path.join(self.data_root, f"body{cfg.body_suffix}_sr_cache", f'v_{i}.npy')):
            overts = np.load(os.path.join(self.data_root, f"body{cfg.body_suffix}_sr_cache", f'v_{i}.npy'))
            faces = np.load(os.path.join(self.data_root, f"body{cfg.body_suffix}_sr_cache", f'f_{i}.npy'))
        else:
            overts, faces = self.load_obj(obj_path)
            os.makedirs(os.path.join(self.data_root, f"body{cfg.body_suffix}_sr_cache"), exist_ok=True)
            np.save(os.path.join(self.data_root, f"body{cfg.body_suffix}_sr_cache", f'v_{i}.npy'), overts)
            np.save(os.path.join(self.data_root, f"body{cfg.body_suffix}_sr_cache", f'f_{i}.npy'), faces)
        # coords system switch: from( x: right, y: up, z: front) to (x: right, y: back, z: up)
        wverts = overts[:, [0, 2, 1]]
        wverts[:, 1] *= -1
        wtrans = self.load_trans(offset_path)
        otrans = wtrans[[0, 2, 1]]
        otrans[2] *= -1
        sverts = wverts + wtrans.reshape(1,3)


        return faces, sverts


    def get_curr_frame_data(self, index):
        if cfg.sdf_input.rgb_gt:
            ref_img_lst = []
            ref_msk_lst = []
            ref_orig_msk_lst = []
            for referece_idx in range(len(cfg.reference_view)):
                #C:\Users\D-Blue\Desktop\cloth\local\surface-aligned-nerf-neus2-nooffset-coarse-icon\d
                ref_img_path = os.path.join(self.data_root, self.reference_ims[referece_idx][index].replace("\\","/").replace('images',self.img_src))
                ref_img = imageio.imread(ref_img_path).astype(np.float32) / 255.
                ref_msk, ref_orig_msk = self.get_mask(self.reference_ims[referece_idx][index])
                ref_img_lst.append(ref_img)
                ref_msk_lst.append(ref_msk)
                ref_orig_msk_lst.append(ref_orig_msk)

        if cfg.is_ref_finetune:
            img_path = os.path.join(self.data_root, self.ims[index].replace("\\","/").replace('images',self.img_src))
        else:
            img_path = os.path.join(self.data_root, self.ims[index].replace("\\","/"))
        # print(img_path)
        img = imageio.imread(img_path).astype(np.float32) / 255.

        # img = cv2.resize(img, (cfg.W, cfg.H), interpolation=cv2.INTER_NEAREST)

        # msk, orig_msk = self.get_mask(index)
        msk, orig_msk = self.get_mask(self.ims[index])
        # if cfg.normal_constraint:
        # normal = self.get_normal(index)
        # H, W = img.shape[:2]
        # msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
        # orig_msk = cv2.resize(orig_msk, (W, H), interpolation=cv2.INTER_NEAREST)



        cam_ind = self.cam_inds[index]
        K = np.array(self.cams['K'][cam_ind])
        D = np.array(self.cams['D'][cam_ind])
        if cfg.distort:
            img = cv2.undistort(img, K, D)
            msk = cv2.undistort(msk, K, D)
            orig_msk = cv2.undistort(orig_msk, K, D)
            # if cfg.normal_constraint:
            # normal = cv2.undistort(normal, K, D)
            if cfg.sdf_input.rgb_gt:
                for referece_idx in range(len(cfg.reference_view)):
                    ref_img_lst[referece_idx] = cv2.undistort(ref_img_lst[referece_idx], self.reference_cams[referece_idx]['K'], self.reference_cams[referece_idx]['D'])
                    ref_msk_lst[referece_idx] = cv2.undistort(ref_msk_lst[referece_idx], self.reference_cams[referece_idx]['K'], self.reference_cams[referece_idx]['D'])
                    ref_orig_msk_lst[referece_idx] = cv2.undistort(ref_orig_msk_lst[referece_idx], self.reference_cams[referece_idx]['K'], self.reference_cams[referece_idx]['D'])


        R = np.array(self.cams['R'][cam_ind])
        if cfg.rotate_camera:
            add_degree = -6 * (index // self.num_cams - cfg.prev_frames)
            R = np.dot(R, cv2.Rodrigues(np.array([0, 0, np.deg2rad(add_degree)]))[0])
        if cfg.novel_pose_rotation:
            x_rot,y_rot,z_rot = cfg.novel_pose_rotation
            R = np.dot(R, cv2.Rodrigues(np.array([np.deg2rad(x_rot), np.deg2rad(y_rot), np.deg2rad(z_rot)]))[0])
        cam_R = R

        T = np.array(self.cams['T'][cam_ind]) / 1000.

        # reduce the image resolution by ratio
        H, W = int(cfg.H * cfg.ratio), int(cfg.W * cfg.ratio)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
        orig_msk = cv2.resize(orig_msk, (W, H), interpolation=cv2.INTER_NEAREST)
        if cfg.sdf_input.rgb_gt:
            for referece_idx in range(len(cfg.reference_view)):
                ref_img_lst[referece_idx] = cv2.resize(ref_img_lst[referece_idx], (W, H), interpolation=cv2.INTER_AREA)
                ref_msk_lst[referece_idx] = cv2.resize(ref_msk_lst[referece_idx], (W, H), interpolation=cv2.INTER_NEAREST)
                ref_orig_msk_lst[referece_idx] = cv2.resize(ref_orig_msk_lst[referece_idx], (W, H), interpolation=cv2.INTER_NEAREST)

        if cfg.mask_bkgd:
            img[msk == 0] = 0
            if cfg.sdf_input.rgb_gt and not 'ref' in self.img_src:
                for referece_idx in range(len(cfg.reference_view)):
                    ref_img_lst[referece_idx][ref_msk_lst[referece_idx] == 0] = 0
            if cfg.white_bkgd:
                img[msk == 0] = 1
                if cfg.sdf_input.rgb_gt and not 'ref' in self.img_src:
                    for referece_idx in range(len(cfg.reference_view)):
                        ref_img_lst[referece_idx][ref_msk_lst[referece_idx] == 0] = 1

        K[:2] = K[:2] * cfg.ratio

        if self.human in ['CoreView_313', 'CoreView_315']:
            i = int(os.path.basename(img_path).split('_')[4])
            frame_index = i - 1
        else:
            i = int(os.path.basename(img_path)[:-4])
            frame_index = i


        body_info, _ = self.prepare_input(i)
        sr_faces, sr_sverts = self.prepare_sr_input(i)
        # coarse_info = self.prepare_coarse_input(i)

        sbounds = if_nerf_data_utils.get_bounds(body_info['sverts'])
        wbounds = if_nerf_data_utils.get_bounds(body_info['wverts'])
        pbounds = if_nerf_data_utils.get_bounds(body_info['pverts'])

        if cfg.sample_bound:# and self.split == 'train':
            rgb, ray_o, ray_d, near, far, coord, mask_at_box, patch_info = if_nerf_data_utils.sample_ray_h36m_bbox(
                img, msk, K, R, T, sbounds, self.nrays, self.split)
        else:
            if cfg.grid_sample and self.split == 'train':
                rgb, ray_o, ray_d, near, far, coord, mask_at_box, patch_info = if_nerf_data_utils.sample_ray_h36m_grid(
                    img, msk, K, R, T, sbounds, self.nrays, self.split, self.grid_sampler)
            elif cfg.neural_rendering :
                if cfg.patch_sample:
                    rgb, ray_o, ray_d, near, far, coord, mask_at_box, patch_info = if_nerf_data_utils.sample_ray_h36m(
                    img, msk, K, R, T, sbounds, self.nrays, self.split, use_full_patch=True)
                else:
                    rgb, ray_o, ray_d, near, far, coord, mask_at_box, patch_info = if_nerf_data_utils.sample_ray_h36m_full(
                    img, msk, K, R, T, sbounds, self.nrays, self.split)
            else:
                rgb, ray_o, ray_d, near, far, coord, mask_at_box, patch_info = if_nerf_data_utils.sample_ray_h36m(
                    img, msk, K, R, T, sbounds, self.nrays, self.split)



        if cfg.erode_edge:
            orig_msk = if_nerf_data_utils.crop_mask_edge(orig_msk)

        mask_at_box_real = if_nerf_data_utils.get_mask_at_box(img, msk, K, R, T, sbounds)

        # nerf and sdf


        ret = {
            'sr_faces': sr_faces,
            'sr_sverts': sr_sverts,
            'mask_at_box_real': mask_at_box_real,
            'msk_2d': orig_msk,
            'img_2d': img,
            'data_index':i,
            # 'select_coord': coord, # [1024, 2]
            'rgb': rgb,

            'ray_o': ray_o,# [1024, 3]
            'ray_d': ray_d,
            'near': near,
            'far': far,
            # 'mask_at_box': mask_at_box,
        }
        ret['select_coord'] = coord
        ret['mask_at_box'] = mask_at_box
        if cfg.neural_rendering and not cfg.patch_sample:
            res = np.sqrt(self.nrays).astype(np.int32)
            orig_msk = orig_msk[::H//res, ::W//res]
        occupancy = orig_msk[coord[:, 0], coord[:, 1]]
        ret['occupancy'] = occupancy

        # reference input
        if cfg.sdf_input.rgb_gt:
            ret['ref_img_2d'] = ref_img_lst
            ret['ref_R'] = [np.array(self.reference_cams[referece_idx]['R']) for referece_idx in range(len(cfg.reference_view))]
            ret['ref_T'] = [np.array(self.reference_cams[referece_idx]['T'])  for referece_idx in range(len(cfg.reference_view))]
            ret['ref_K'] = [np.array(self.reference_cams[referece_idx]['K']) for referece_idx in range(len(cfg.reference_view))]
        else:
            ret['ref_img_2d'] = [img]
        # blend weight
        meta = {
            'sbounds': sbounds,
            'wbounds': wbounds,
            'overts':body_info['overts'],
            'wverts': body_info['wverts'],
            'wtrans': body_info['wtrans'],
            'sverts': body_info['sverts'],
            'faces':body_info['faces'],
            # 'coarse_wverts': coarse_info['wverts'],
            # 'coarse_sverts': coarse_info['sverts'],
            # 'coarse_pverts': coarse_info['pverts'],
            # 'coarse_faces':coarse_info['faces'],
        }
        meta.update({
            'pbounds': pbounds,
            'pverts': body_info['pverts'],
            'prot': body_info['prot'],
            'ptrans': body_info['ptrans'],
        })
        ret.update(meta)

        # transformation
        meta = {'H': H, 'W': W}
        ret.update(meta)

        latent_index = index // self.num_cams

        if cfg.test_novel_pose:
            # todo:这个感觉不太对啊，latent_index是表示帧数的，也就是当前的时间，而cfg.num_train_frame是表示训练的总帧数
            latent_index = cfg.num_train_frame - 1
        # meta
        ret.update({
            # latent index表示当前帧在当前split中的帧数
            # frame_index表示当前帧在真实数据集中的帧数
            'latent_index': latent_index - cfg.prev_frames,# todo:这里会导致cuda error
            'frame_index': frame_index,
            'cam_ind': cam_ind,
            'extra_rot': index // self.num_cams - cfg.prev_frames
        })
        # joints
        ret.update({
            'img_path': img_path,
            'cam_R': cam_R,
            'b_faces': body_info['faces'],

        })
        ret.update({
            'wjoints': body_info['wjoints'],
            'sjoints': body_info['sjoints'],
            'pjoints': body_info['pjoints'],
        })
        if cfg.patch_sample:
            ret.update(patch_info)
        return ret

    def get_frame_vel(self, index, prev_n):
        index = index - prev_n * self.num_cams

        img_path = os.path.join(self.data_root, self.ims[index].replace("\\","/"))
        i = int(os.path.basename(img_path)[:-4])
        body_info, params = self.prepare_input(i)
        # coarse_info = self.prepare_coarse_input(i)
        ret = {f'wverts_prev_{prev_n}': body_info['wverts'],
               f'wtrans_prev_{prev_n}': body_info['wtrans'],
               f'sverts_prev_{prev_n}': body_info['sverts'],
               # f'coarse_wverts_prev_{prev_n}': coarse_info['wverts'],
               # f'coarse_wtrans_prev_{prev_n}': coarse_info['wtrans'],
               # f'coarse_sverts_prev_{prev_n}': coarse_info['sverts'],

           }
        ret.update({
                f'pverts_prev_{prev_n}': body_info['pverts'],
                f'wjoints_prev{prev_n}': body_info['wjoints'],
                f'sjoints_prev{prev_n}': body_info['sjoints'],
                f'pjoints_prev{prev_n}': body_info['pjoints'],
            })
        return ret, params

    def __getitem__(self, index):
        # time windows = prev_frames + 1
        # real_index是time windows最后一帧，而输入的index是time windows的第一帧
        real_index = index + cfg.prev_frames * self.num_cams
        # frame_idx = real_index // self.num_cams
        ret = self.get_curr_frame_data(real_index)

        if cfg.sdf_input.body_velocity:
            for i in range(cfg.prev_frames): #(0~cfg.prev_frames-1)
                vel_data, params = self.get_frame_vel(real_index, i + 1)
                curr_wverts = vel_data[f'wverts_prev_{i+1}']
                if i == 0:
                    next_wverts = ret['wverts']
                else:
                    next_wverts = ret[f'wverts_prev_{i}']
                vel_data[f'cano_vel_prev_{i+1}'] = self.body_model.canonnicalize(torch.tensor(next_wverts-curr_wverts),
                                                                               betas=params['betas'].detach().cpu(),
                                                                               body_pose=params['body_pose'].detach().cpu(),
                                                                               global_orient=params['global_orient'].detach().cpu(),
                                                                               is_offset=True)


                ret.update(vel_data)

        return ret

    def __len__(self):
        return len(self.ims) - self.num_cams * cfg.prev_frames