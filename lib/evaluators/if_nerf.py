import numpy as np
from lib.config import cfg
# from skimage.measure import compare_ssim
from skimage.metrics import structural_similarity as compare_ssim
import os
import cv2
from termcolor import colored
import torch
import trimesh
class Evaluator:
    def __init__(self):
        self.mse = []
        self.psnr = []
        self.ssim = []

    def psnr_metric(self, img_pred, img_gt):
        mse = np.mean((img_pred - img_gt)**2)
        psnr = -10 * np.log(mse) / np.log(10)
        return psnr

    def ssim_metric(self, img_pred, img_gt, batch, output, mask_pred=None):
        if 'iter_step' in batch.keys():
            prefix = f"iter{batch['iter_step']:06d}_"
        elif 'prefix' in batch.keys():
            prefix = batch['prefix']
        else:
            raise ValueError('Either "prefix" or "iter_step" need to be provided in batch.')


        result_dir = cfg.result_dir
        
        # os.system('mkdir -p {}'.format(result_dir))
        os.makedirs(result_dir, exist_ok=True)
        frame_index = batch['frame_index'].item()
        view_index = batch['cam_ind'].item()
        if 'posed' in cfg.test_dataset.data_root or  'mask_at_box_real' not in batch:
            mask_bbox = np.ones((512, 512))
        else:
            mask_bbox = batch['mask_at_box_real'].cpu().numpy()
        H, W = int(cfg.H * cfg.ratio), int(cfg.W * cfg.ratio)
        mask_bbox = mask_bbox.reshape(H, W)
        # crop the object region
        x, y, w, h = cv2.boundingRect(mask_bbox.astype(np.uint8))
        np.save(f'{result_dir}/{prefix}frame{frame_index:04d}_view{view_index:04d}.npy', np.array([x, y, w, h]))
        if mask_pred is not None:
            img_pred = img_pred * mask_pred
        cv2.imwrite(
            f'{result_dir}/{prefix}frame{frame_index:04d}_view{view_index:04d}_ref.png',
            (batch['ref_img_2d'][0][0,..., [2, 1, 0]].cpu().numpy() * 255))
        cv2.imwrite(
            f'{result_dir}/{prefix}frame{frame_index:04d}_view{view_index:04d}.png',
            (img_pred[..., [2, 1, 0]] * 255))
        cv2.imwrite(
            f'{result_dir}/{prefix}frame{frame_index:04d}_view{view_index:04d}_trans.png',
            np.concatenate([img_pred,mask_pred],axis=-1)[...,[2,1,0,3]]*255)
        cv2.imwrite(
            f'{result_dir}/{prefix}frame{frame_index:04d}_view{view_index:04d}_gt.png',
            (img_gt[..., [2, 1, 0]] * 255))
        # compute the ssim
        ssim = compare_ssim(img_pred, img_gt, multichannel=True)
        return ssim

    def plot_normal(self, output, batch):
        no_intersect_mask = output['no_intersect_mask']
        if 'iter_step' in batch.keys():
            prefix = f"iter{batch['iter_step']:06d}_"
        elif 'prefix' in batch.keys():
            prefix = batch['prefix']
        else:
            raise ValueError('Either "prefix" or "iter_step" need to be provided in batch.')
        mask_at_box = batch['mask_at_box'][0].detach().cpu().numpy()
        H, W = int(cfg.H * cfg.ratio), int(cfg.W * cfg.ratio)
        mask_at_box = mask_at_box.reshape(H, W)


        select_coord = (batch['select_coord']/cfg.ratio).long()#[1, 512, 2]
        normal_map = batch['normal'] #[1, 1024, 1024, 3]
        if not cfg.use_smpl:
            normal_map[(torch.round(normal_map,decimals=6)==0.003922).sum(3)==3] = -1.
        normal_gt = normal_map[0,select_coord[..., 0], select_coord[..., 1], :].cpu() #[1, 512, 3]
        normal_mask = (normal_gt.sum(2)!=-3.) & no_intersect_mask.cpu()
        normal_gt = (normal_gt/normal_gt.norm(dim=2,keepdim=True))
        white_bkgd = int(cfg.white_bkgd)
        H, W = int(cfg.H * cfg.ratio), int(cfg.W * cfg.ratio)
        normal_gt_img = np.zeros((H, W, 3)) + white_bkgd
        normal_gt[~normal_mask] = -1
        normal_gt_img[mask_at_box] = normal_gt

        normal_pred = (batch['cam_R'][0].float() @ output['surface_normal'][0].cuda().T).T.detach().cpu()[None]
        # normal_pred = normal_pred[normal_mask]
        normal_pred = normal_pred/normal_pred.norm(dim=2,keepdim=True)#[ :, [0,2,1]]
        normal_pred[...,1] *= -1
        normal_pred[...,2] *= -1

        white_bkgd = int(cfg.white_bkgd)
        normal_pred_img = np.zeros((H, W, 3)) + white_bkgd
        normal_pred[~normal_mask] = -1
        normal_pred_img[mask_at_box] = normal_pred
        # res = np.concatenate((normal_gt_img,normal_pred_img),axis=1)
        # plt.imshow(res)
        # plt.show()

        result_dir = cfg.result_dir
        os.makedirs(result_dir, exist_ok=True)
        frame_index = batch['frame_index'].item()
        view_index = batch['cam_ind'].item()
        if batch["extra_rot"] != -1:
            cv2.imwrite(
                f'{result_dir}/{prefix}frame{frame_index:04d}_view{view_index:04d}_rot{batch["extra_rot"].item():04d}_normal.png',
                (normal_pred_img[..., [2, 1, 0]] * 255))
            cv2.imwrite(
                f'{result_dir}/{prefix}frame{frame_index:04d}_view{view_index:04d}_rot{batch["extra_rot"].item():04d}_normal_gt.png',
                (normal_gt_img[..., [2, 1, 0]] * 255))
            # self.save_obj(f'{result_dir}/{prefix}frame{frame_index:04d}_view{view_index:04d}_normal_gt.obj',
            #               batch['wverts'][0], batch['vt'], batch['faces'][0], batch['ft'], batch['tex_path'])
        else:
            cv2.imwrite(
                f'{result_dir}/{prefix}frame{frame_index:04d}_view{view_index:04d}_normal.png',
                (normal_pred_img[..., [2, 1, 0]] * 255))
            cv2.imwrite(
                f'{result_dir}/{prefix}frame{frame_index:04d}_view{view_index:04d}_normal_gt.png',
                (normal_gt_img[..., [2, 1, 0]] * 255))


    def save_obj(self, filename, verts, vert_tex, faces, face_tex, tex_path):
        assert verts.ndimension() == 2
        assert faces.ndimension() == 2

        filename_mtl = filename[:-4] + '.mtl'
        filename_texture = tex_path
        material_name = 'material_1'

        faces = faces.detach().cpu().numpy()

        with open(filename, 'w') as f:
            f.write('# %s\n' % os.path.basename(filename))
            f.write('#\n')
            f.write('\n')

            f.write('mtllib %s\n\n' % os.path.basename(filename_mtl))

            for vertex in verts:
                f.write('v %.8f %.8f %.8f\n' % (vertex[0],  vertex[2],-vertex[1]))
            f.write('\n')


            for vertex in vert_tex:
                f.write('vt %.8f %.8f\n' % (vertex[1], 1-vertex[0]))
            f.write('\n')

            f.write('usemtl %s\n' % material_name)
            for i, face in enumerate(faces):
                f.write('f %d/%d %d/%d %d/%d\n' % (
                    face[0] + 1,face_tex[i,0] + 1, face[1] + 1, face_tex[i,1] + 1, face[2] + 1, face_tex[i,2] + 1))
                # print('f %d/%d %d/%d %d/%d\n' % (
                #     face[0] + 1,face_tex[i,0] + 1, face[1] + 1, face_tex[i,1] + 1, face[2] + 1, face_tex[i,2] + 1))
            f.write('\n')


        with open(filename_mtl, 'w') as f:
            f.write('newmtl %s\n' % material_name)
            f.write('map_Kd %s\n' % os.path.basename(filename_texture))




    def evaluate(self, output, batch):
        # if cfg.normal_constraint:
        # if not cfg.neural_rendering:
        #     self.plot_normal(output, batch)



        # convert the pixels into an image
        # if cfg.neural_rendering:
        #     img_pred = (output['rgb_map'][...,:3]*output['rgb_map'][...,[-1]]).detach().cpu().numpy()
        #     img_gt = batch['img_2d'][0].cpu().numpy()
        # else:
        rgb_pred_ori = output['rgb_map'][0].detach().cpu().numpy()
        if 'rgb' in batch:
            rgb_gt = batch['rgb'][0].detach().cpu().numpy()
        if 'acc_map' in output:
            mask_pred = output['acc_map'][0].detach().cpu().numpy()
        if cfg.neural_rendering and not cfg.patch_sample:
            if not 'posed' in cfg.test_dataset.data_root and 'img_2d' in batch:
                rgb_gt = batch['img_2d'][...,:3].reshape(512,512,3).detach().cpu().numpy()
            else:
                rgb_gt = np.zeros((512,512,3))
            # rgb_pred = (rgb_pred[...,:3]*rgb_pred[...,-1][...,None]).reshape(512,512,3)
            rgb_pred = rgb_pred_ori[...,:3].reshape(512,512,3)
        else:
            # mask_at_box = batch['mask_at_box'][0].detach().cpu().numpy()
            select_coords = batch['select_coord'][0].detach().cpu().numpy()
            H, W = int(cfg.H * cfg.ratio), int(cfg.W * cfg.ratio)
            # mask_at_box = mask_at_box.reshape(H, W)
            white_bkgd = int(cfg.white_bkgd)
            img_pred = np.zeros((H, W, 3)) + white_bkgd
            img_pred[select_coords[:,0],select_coords[:,1]] = rgb_pred_ori[...,:3]
            if cfg.patch_sample and cfg.neural_rendering:
                img_pred[select_coords[:,0],select_coords[:,1]]*=mask_pred[...,None]
            img_gt = np.zeros((H, W, 3)) + white_bkgd
            img_gt[select_coords[:,0],select_coords[:,1]] = rgb_gt
            rgb_pred = img_pred
            rgb_gt = img_gt

            # if cfg.eval_whole_img:
            #     rgb_pred = img_pred
            #     rgb_gt = img_gt

        mse = np.mean((rgb_pred - rgb_gt)**2)
        self.mse.append(mse)

        psnr = self.psnr_metric(rgb_pred, rgb_gt)
        self.psnr.append(psnr)

        if cfg.neural_rendering and not cfg.patch_sample:
            ssim = self.ssim_metric(rgb_pred, rgb_gt, batch, output, mask_pred=rgb_pred_ori[...,-1].reshape(512, 512, 1))
        else:
            ssim = self.ssim_metric(rgb_pred, rgb_gt, batch, output)
        self.ssim.append(ssim)

    def summarize(self):
        result_dir = cfg.result_dir
        print(
            colored('the metrics results are saved at {}'.format(result_dir),
                    'yellow'))

        result_path = os.path.join(cfg.result_dir, 'metrics.npy')
        # os.system('mkdir -p {}'.format(os.path.dirname(result_path)))
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        metrics = {'mse': self.mse, 'psnr': self.psnr, 'ssim': self.ssim}
        np.save(result_path, metrics)
        print('mse: {}'.format(np.mean(self.mse)))
        print('psnr: {}'.format(np.mean(self.psnr)))
        print('ssim: {}'.format(np.mean(self.ssim)))
        ret = {'mse': np.mean(self.mse), 'psnr': np.mean(self.psnr), 'ssim': np.mean(self.ssim)}
        self.mse = []
        self.psnr = []
        self.ssim = []
        return ret
