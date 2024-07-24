import torch.nn as nn
from lib.config import cfg
import torch
from lib.networks.renderer import make_renderer
from lib.train import make_optimizer
import torch.nn.functional as F
from lib.networks.perceptual_loss import PNetLin
class NetworkWrapper(nn.Module):
    def __init__(self, net):
        super(NetworkWrapper, self).__init__()

        self.net = net
        # self.renderer = nerf_renderer.Renderer(self.net)
        self.renderer = make_renderer(cfg, self.net)

        self.img2mse = lambda x, y : ((x - y) ** 2).mean()
        self.acc_crit = torch.nn.functional.smooth_l1_loss
        self.vgg_loss_fn = PNetLin(pnet_tune=False, spatial=False)

    def huber(self, x, y, scaling=0.1):
        """
        A helper function for evaluating the smooth L1 (huber) loss
        between the rendered silhouettes and colors.
        """
        # import ipdb; ipdb.set_trace()
        diff_sq = (x - y) ** 2
        loss = ((1 + diff_sq / (scaling**2)).clamp(1e-4).sqrt() - 1) * float(scaling)
        loss = loss.abs().mean()
        return loss

    def forward(self, batch):
        ret = self.renderer.render(batch)

        scalar_stats = {}
        loss = 0
        n_pixels = cfg.N_rand
        #

        # loss after nerf
        # nerf mask loss

        # loss after neural rendering
        nr_rgb_pred = ret['rgb_map'][..., :3]
        nr_mask_pred = ret['rgb_map'][..., -1, None]
        nr_rgb_gt = batch['img_2d'][...,:3].reshape(1,-1,3)

        nr_cloth_mask_gt = (batch['msk_2d'][0]==cfg.cloth_seg).float().reshape(1,-1,1)
        nr_body_mask_gt = (batch['msk_2d'][0]>0.).float().reshape(1,-1,1)

        # neural rendering mask loss
        if cfg.surface_rendering_v3:
            nr_mask_loss = self.huber(nr_mask_pred, nr_cloth_mask_gt)
        else:
            nr_mask_loss = self.huber(nr_mask_pred, nr_body_mask_gt)
        scalar_stats.update({'nr_mask_loss': nr_mask_loss})
        loss += cfg.msk_wt * nr_mask_loss

        # rgb loss

        img_loss = self.huber(nr_rgb_pred[:,nr_body_mask_gt.squeeze().bool()], nr_rgb_gt[:,nr_body_mask_gt.squeeze().bool()])

        scalar_stats.update({'img_loss': img_loss})
        loss +=cfg.img_wt * img_loss

        # post process
        if cfg.surface_rendering_v3:
            sr_body_rgb = batch['body_rgb']
            ret['rgb_map'] = nr_rgb_pred * nr_mask_pred + (1-nr_mask_pred)*sr_body_rgb
        else:
            ret['rgb_map'] = nr_rgb_pred * nr_mask_pred


        if cfg.perceptual_wt > 0:
            vgg_loss = self.vgg_loss_fn((nr_rgb_gt*nr_body_mask_gt).reshape(512,512,3).permute(2, 0, 1)[None],
                                        (nr_rgb_pred*nr_body_mask_gt).reshape(512,512,3).permute(2, 0, 1)[None]) * cfg.perceptual_wt
            scalar_stats.update({'vgg_loss': vgg_loss})
            loss += vgg_loss


        scalar_stats.update({'loss': loss})

        image_stats = {    'rgb': batch['img_2d'],
                           'rgb_map': ret['rgb_map'][...,:3].reshape(512,512,3)
                       }

        if cfg.neural_rendering:
            image_stats['mask_at_box'] = batch['mask_at_box'][0]
        return ret, loss, scalar_stats, image_stats


