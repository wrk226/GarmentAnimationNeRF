import matplotlib.pyplot as plt
import torch.nn as nn
from lib.config import cfg
import torch
from lib.networks.renderer import make_renderer
from lib.visualizers import make_visualizer
from lib.train import make_optimizer
from lib.utils.img_utils import get_img
import time
import psutil
from . import crit
import datetime

import torch.nn.functional as F
from lib.networks.perceptual_loss import PNetLin
class NetworkWrapper(nn.Module):
    def __init__(self, net):
        super(NetworkWrapper, self).__init__()

        self.net = net
        self.renderer = make_renderer(cfg, self.net)


        self.acc_crit = torch.nn.functional.smooth_l1_loss
        self.img2mse = lambda x, y : torch.mean((x - y) ** 2)
        self.vgg_loss_fn = PNetLin(pnet_tune=False, spatial=False)

    def huber(self, x, y, scaling=0.1):
        """
        A helper function for evaluating the smooth L1 (huber) loss
        between the rendered silhouettes and colors.
        """
        # import ipdb; ipdb.set_trace()
        diff_sq = (x - y) ** 2
        loss = ((1 + diff_sq / (scaling**2)).clamp(1e-4).sqrt() - 1) * float(scaling)
        # if mask is not None:
        #     loss = loss.abs().sum()/mask.sum()
        # else:
        loss = loss.abs().mean()
        return loss

    def forward(self, batch):
        ret = self.renderer.render(batch)
        scalar_stats = {}
        loss = 0
        if 'gradients' in ret:
            gradients = ret['gradients'].clone()
            grad_loss = (torch.norm(gradients, dim=2) - 1.0)**2
            grad_loss = grad_loss.mean()
            scalar_stats.update({'grad_loss': grad_loss})
            if cfg.gradient_wt > 0:
                loss += cfg.gradient_wt * grad_loss

        body_mask_gt = (batch['occupancy'].float()>0.).float()
        body_mask_pred = ret['acc_map']
        if cfg.use_msk_loss:

            mask_loss = self.huber(body_mask_pred, body_mask_gt)

            scalar_stats.update({'mask_loss': mask_loss})
            if cfg.msk_wt > 0:
                loss += cfg.msk_wt * mask_loss


        if cfg.normal_wt > 0 and cfg.train_body_sdf:
            pass
        no_intersect_mask = ret['no_intersect_mask']
        select_coord = (batch['select_coord']/cfg.ratio).long()#[1, 512, 2]

        if cfg.normal_wt > 0:
            # todo：对不同视角加weight，因为normal可能预测不准确
            normal_map = batch['normal'] #[1, 1024, 1024, 3]
            if not cfg.use_smpl:
                normal_map[(torch.round(normal_map,decimals=6)==0.003922).sum(3)==3] = -1.
                # import matplotlib.pyplot as plt
                # plt.imshow(normal_map[0].cpu().sum(2)==-3.)
                # plt.show()

            normal_gt = normal_map[0, select_coord[..., 0], select_coord[..., 1], :] #[1, 512, 3]
            normal_mask = (normal_gt.sum(2)!=-3.)&no_intersect_mask


            # normal_mask = batch['occupancy']>=1
            normal_gt = normal_gt[normal_mask]
            # normal_gt = (normal_gt/normal_gt.norm(dim=1,keepdim=True))

            normal_pred = ret['surface_normal'].clone()
            # normal_pred = normal_pred @ torch.inverse(batch['cam_R'].float())
            normal_pred = (batch['cam_R'][0].float() @ normal_pred[0].T).T[None]
            normal_pred = normal_pred[normal_mask]
            # normal_pred = normal_pred/normal_pred.norm(dim=1,keepdim=True)#[ :, [0,2,1]]
            normal_pred[:,1] *= -1
            normal_pred[:,2] *= -1

            normals_loss = self.huber(normal_pred, normal_gt)


            # import matplotlib.pyplot as plt
            # fig = plt.figure()
            # ax1 = fig.add_subplot(1, 2, 1)
            # ax1.scatter(select_coord[..., 0][normal_mask].cpu(), select_coord[..., 1][normal_mask].cpu(), c=((normal_gt.detach().cpu()+1)/2))
            # ax1.set_title('Ground Truth')
            # ax2 = fig.add_subplot(1, 2, 2)
            # ax2.scatter(select_coord[..., 0][normal_mask].cpu(), select_coord[..., 1][normal_mask].cpu(), c=((normal_pred.detach().cpu())+1)/2)
            # ax2.set_title('Prediction')
            # plt.show()

            if not torch.isnan(normals_loss) and cfg.normal_wt > 0:
                scalar_stats.update({'normal_loss': normals_loss})
                loss += normals_loss * cfg.normal_wt


        rgb_pred = ret['rgb_map']
        rgb_gt = batch['rgb']
        img_loss = self.huber(rgb_pred[:,body_mask_gt[0].bool()],rgb_gt[:,body_mask_gt[0].bool()])
        scalar_stats.update({'img_loss': img_loss})
        if cfg.img_wt > 0:
            loss += img_loss * cfg.img_wt

        scalar_stats.update({'loss': loss})

        image_stats = {'select_coord':batch['select_coord'],
                       'rgb': batch['rgb'],
                       'mask_at_box': batch['mask_at_box'],
                       'rgb_map': ret['rgb_map'],
                        # 'disp_map': ret['disp_map'],
                        'acc_map': ret['acc_map'],
                        'depth_map': ret['depth_map'],
                       }
        return ret, loss, scalar_stats, image_stats


