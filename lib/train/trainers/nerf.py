import torch.nn as nn
from lib.config import cfg
import torch
from lib.networks.renderer import make_renderer
from lib.train import make_optimizer
import torch.nn.functional as F
from lib.networks.perceptual_loss import PNetLin
from lib.networks import vgg_dng
import numpy as np
from scipy.ndimage import label

class NetworkWrapper(nn.Module):
    def __init__(self, net):
        super(NetworkWrapper, self).__init__()

        self.net = net
        # self.renderer = nerf_renderer.Renderer(self.net)
        self.renderer = make_renderer(cfg, self.net)

        self.img2mse = lambda x, y : ((x - y) ** 2).mean()
        self.acc_crit = torch.nn.functional.smooth_l1_loss
        self.vgg_loss_fn = PNetLin(pnet_tune=False, spatial=False)
        self.DataLoss = torch.nn.L1Loss().to('cuda')
        self.vgg = vgg_dng.VGG19('cuda').to('cuda')
        for param in self.vgg.parameters():
            param.requires_grad = False
        from torchvision import transforms
        self.vggImg_transform = transforms.Compose([
                                                    # transforms.ToTensor(),
                                                    transforms.Normalize(vgg_dng.vgg_mean , vgg_dng.vgg_std )])

    def peceptronLoss(self, gt, x, Layers):
        gt = self.vggImg_transform(gt[0])[None]
        x = self.vggImg_transform(x[0])[None]

        dist = 0.
        if len(Layers) == 0:
            return self.DataLoss(gt, x)

        gtFeats = self.vgg.get_content_actList(gt, Layers)
        xxFeats = self.vgg.get_content_actList(x, Layers)

        for l in range(len(Layers)):
            dist += self.DataLoss(gtFeats[l], xxFeats[l])
        dist += self.DataLoss(gt, x)
        return dist

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

    def remove_disconnected_area(self, weight_map):
        # 将Tensor转换为NumPy数组并进行二值化
        input_np = weight_map.squeeze()  # 去除单一维度
        binary_np = (input_np > 0.5).astype(np.int)  # 设置阈值为0.5进行二值化

        # 使用scipy进行连通组件分析
        labeled_array, num_features = label(binary_np)

        # 找到最大连通组件
        max_area = 0
        max_label = 0
        for region in range(1, num_features + 1):
            area = (labeled_array == region).sum()
            if area > max_area:
                max_area = area
                max_label = region

        # 创建一个只包含最大连通组件的新数组
        output_np = np.where(labeled_array == max_label, input_np, 0)

        # 将结果转换回PyTorch Tensor
        output_tensor = torch.tensor(output_np, dtype=torch.float32)
        return output_tensor

    def forward(self, batch):
        ret = self.renderer.render(batch)

        scalar_stats = {}
        loss = 0
        n_pixels = cfg.N_rand


        # loss after nerf
        # nerf mask loss
        body_mask_gt = (batch['occupancy'].float()>0.).float()

        body_mask_pred = ret['acc_map']
        acc_mask_loss = self.huber(body_mask_pred, body_mask_gt)
        scalar_stats.update({'acc_mask_loss': cfg.nerf_msk_wt * acc_mask_loss})
        loss += cfg.nerf_msk_wt * acc_mask_loss
        if cfg.neural_rendering:
            # loss after neural rendering
            # nr_rgb_pred = ret['basis_rgb'].mean(dim=-2)
            nr_rgb_pred = ret['rgb_map'][..., :3]
            nr_mask_pred = ret['rgb_map'][..., -1, None]

            nr_rgb_gt = batch['img_2d'][...,:3].reshape(1,-1,3)
            bounding_mask_gt = batch['mask_at_box']

            nr_cloth_mask_gt = (batch['msk_2d'][0]==cfg.cloth_seg).float().reshape(1,-1,1)
            nr_body_mask_gt = (batch['msk_2d'][0]>0.).float().reshape(1,-1,1)

            # loss for palette
            # palette loss
            if cfg.palette_wt>0:
                palette_loss = ((self.net.basis_color - self.net.basis_color_ori)**2).sum(dim=-1).mean()
                scalar_stats.update({'palette_loss': cfg.palette_wt * palette_loss})
                loss += cfg.palette_wt * palette_loss
            # omega sparsity
            if cfg.omega_sparsity_wt>0:
                omega = ret['omega_map']
                omega_sparsity = omega.squeeze().sum(dim=-1, keepdim=True) / ((omega.squeeze()**2).sum(dim=-1, keepdim=True)+1e-6)-1
                # omega_sparsity = omega_sparsity.mean()
                omega_sparsity = omega_sparsity[nr_body_mask_gt.squeeze().bool()].mean()
                scalar_stats.update({'omega_sparsity': cfg.omega_sparsity_wt * omega_sparsity})
                loss += cfg.omega_sparsity_wt * omega_sparsity
            # offsets_norm
            if cfg.offsets_norm_wt>0:
                offsets = ret['offset_map']
                offsets_norm = (offsets.squeeze()**2).sum(dim=-1).sum(dim=-1, keepdim=True) # (N_rays, N_samples_, 1)
                # offsets_norm = offsets_norm.mean()#offsets_norm[nr_body_mask_gt.squeeze().bool()].mean()
                offsets_norm = offsets_norm[nr_body_mask_gt.squeeze().bool()].mean()
                scalar_stats.update({'offsets_norm': cfg.offsets_norm_wt * offsets_norm})
                loss += cfg.offsets_norm_wt * offsets_norm
            if cfg.sim_wt>0:
                #[:,nr_body_mask_gt.squeeze().bool()]
                rgb_gt = nr_rgb_gt
                rgb_gt_v100 = rgb_gt/rgb_gt.max(dim=-1, keepdim=True).values
                basis_color_v100 = self.renderer.net.basis_color/self.renderer.net.basis_color.max(dim=-1, keepdim=True).values
                cal_offset = (rgb_gt_v100[:,:,None] - basis_color_v100[None,None])
                basis_idx = (cal_offset.abs()).sum(-1).min(-1).indices
                basis_label = torch.zeros(512*512, cfg.num_basis,device='cuda')
                basis_label[torch.arange(512*512), basis_idx] = 1

                yy, xx = torch.meshgrid(torch.arange(512), torch.arange(512), indexing='ij')
                xy_offset = ((torch.randn(512,512, 2)*0.3).clamp(-1, 1)*10).int()
                yy = (yy + xy_offset[:,:,0]).clamp(0, 511)
                xx = (xx + xy_offset[:,:,1]).clamp(0, 511)
                basis_label_diff = (basis_label*nr_body_mask_gt[0]).reshape(512,512,cfg.num_basis)[yy, xx].view(512*512, cfg.num_basis)
                basis_label_refine = basis_label * basis_label_diff
                omega = ret['omega_map']
                # sim_loss = nn.BCELoss()(omega.reshape(512*512,4), basis_label_refine.cuda())
                targets_indices = torch.argmax(basis_label_refine, dim=1).cuda()
                sim_loss = F.cross_entropy(omega.reshape(512*512,cfg.num_basis)[nr_body_mask_gt.squeeze().bool()], targets_indices[nr_body_mask_gt.squeeze().bool()])
                scalar_stats.update({'sim_loss': cfg.sim_wt * sim_loss})
                loss += cfg.sim_wt * sim_loss



            # basis color
            if cfg.basis_color_wt>0:
                palette_loss = ((self.renderer.net.basis_color - self.renderer.net.basis_color_ori)**2).sum(dim=-1).mean()
                scalar_stats.update({'palette_loss': cfg.basis_color_wt * palette_loss})
                loss += cfg.basis_color_wt * palette_loss

            if cfg.balance_wt>0:
                # omega_mean = ret['omega_map'].mean(dim=1).squeeze()
                omega_mean = (ret['omega_map'][:,nr_body_mask_gt.squeeze().bool()]**2).mean(dim=1).squeeze()
                balance_loss = ((omega_mean-omega_mean.mean())**2).mean()
                scalar_stats.update({'balance_loss': cfg.balance_wt * balance_loss})
                loss += cfg.balance_wt * balance_loss

            if cfg.v100_wt>0:
                # offsets = ret['offset_map']
                offsets = ret['offset_map'][:,nr_body_mask_gt.squeeze().bool()]
                basis_color = self.renderer.net.basis_color
                v100_loss = (offsets[:,:,torch.arange(0,cfg.num_basis),basis_color.max(dim=-1).indices.squeeze()]**2).mean()
                scalar_stats.update({'v100_loss': cfg.v100_wt * v100_loss})
                loss += cfg.v100_wt * v100_loss
            if cfg.cd_wt >0:
                cd_loss = 0.
                omega_map_backup = ret['omega_map'].reshape(512,512,cfg.num_basis).detach().cpu().numpy()

                for area in cfg.cd_area:
                    cleaned_omega_map = self.remove_disconnected_area(omega_map_backup[..., area]).flatten().cuda()
                    if cfg.binary_cd:
                        cleaned_omega_map = (cleaned_omega_map>0.).float()
                    cd_loss += self.huber(ret['omega_map'][0,nr_body_mask_gt.squeeze().bool(),area,0], cleaned_omega_map[nr_body_mask_gt.squeeze().bool()], )
                scalar_stats.update({'cd_loss': cfg.cd_wt * cd_loss})
                loss += cfg.cd_wt * cd_loss

            if cfg.smooth_wt>0:
                omega = ret['omega_map'].reshape(512,512,cfg.num_basis)
                offset = ret['offset_map'].reshape(512,512,cfg.num_basis,3)
                yy, xx = torch.meshgrid(torch.arange(512), torch.arange(512), indexing='ij')
                xy_offset = ((torch.randn(512,512, 2)*0.3).clamp(-1, 1)*10).int()
                yy = (yy + xy_offset[:,:,0]).clamp(0, 511)
                xx = (xx + xy_offset[:,:,1]).clamp(0, 511)
                omega_diff = omega[yy, xx]
                offset_diff = offset[yy, xx]
                xy_wt = (xy_offset/512.).to('cuda').norm(dim=-1, keepdim=True)**2 / 0.005
                rgb_wt = (offset_diff - offset).reshape(512,512,-1).norm(dim=-1, keepdim=True)**2 / 0.2
                smooth_wt = torch.exp(- xy_wt - rgb_wt).detach()
                smooth_norm = (((omega_diff-omega)**2).sum(dim=-1, keepdim=True) * smooth_wt).mean()
                scalar_stats.update({'smooth_loss': cfg.smooth_wt * smooth_norm})
                loss += cfg.smooth_wt * smooth_norm

            # if cfg.view_dep_wt >0:
            #     view_dep_norm = (view_dep**2).sum(dim=-1, keepdim=True)

            # neural rendering mask loss
            if cfg.surface_rendering_v3:
                nr_mask_loss = self.huber(nr_mask_pred, nr_cloth_mask_gt)
            else:
                nr_mask_loss = self.huber(nr_mask_pred, nr_body_mask_gt)
            scalar_stats.update({'nr_mask_loss': cfg.nr_msk_wt * nr_mask_loss})
            loss += cfg.nr_msk_wt * nr_mask_loss

            # rgb loss
            if cfg.img_wt > 0:
                img_loss = self.huber(nr_rgb_pred[:,nr_body_mask_gt.squeeze().bool()], nr_rgb_gt[:,nr_body_mask_gt.squeeze().bool()])
                scalar_stats.update({'img_loss': cfg.img_wt * img_loss})
                loss +=cfg.img_wt * img_loss
            if cfg.direct_rgb_wt > 0 and 'direct_rgb_map' in ret:
                nr_direct_rgb_pred = ret['direct_rgb_map']
                direct_img_loss = self.huber(nr_direct_rgb_pred[:,nr_body_mask_gt.squeeze().bool()], nr_rgb_gt[:,nr_body_mask_gt.squeeze().bool()])
                scalar_stats.update({'direct_img_loss': cfg.direct_rgb_wt * direct_img_loss})
                loss +=cfg.direct_rgb_wt * direct_img_loss

            # post process
            ret['rgb_map'] = ret['rgb_map'][..., :3] * ret['rgb_map'][..., -1, None]

            if 'rgb_map_val' in ret:
                ret['rgb_map_val'] = ret['rgb_map_val'] * nr_mask_pred
                ret['basis_rgb'] = ret['basis_rgb'] * nr_mask_pred[...,None]
            if cfg.hard_rgb_decay_ep > -1:
                ret['rgb_map_hard'] = ret['rgb_map_hard'] * nr_mask_pred

        else:
            if cfg.img_wt > 0:
                rgb_pred = ret['rgb_map']
                rgb_gt = batch['rgb']
                img_loss = self.huber(rgb_pred,rgb_gt)
                scalar_stats.update({'img_loss': cfg.img_wt * img_loss})
                loss +=cfg.img_wt * img_loss

        if cfg.perceptual_wt > 0:
            if cfg.vgg_type == 'dng':

                vgg_loss = self.peceptronLoss((nr_rgb_gt*nr_body_mask_gt).reshape(512,512,3).permute(2, 0, 1)[None],
                                              (nr_rgb_pred*nr_body_mask_gt).reshape(512,512,3).permute(2, 0, 1)[None], [4, 12, 30]) * cfg.perceptual_wt
            else:
                vgg_loss = self.vgg_loss_fn((nr_rgb_gt*nr_body_mask_gt).reshape(512,512,3).permute(2, 0, 1)[None],
                                            (nr_rgb_pred*nr_body_mask_gt).reshape(512,512,3).permute(2, 0, 1)[None]) * cfg.perceptual_wt
            scalar_stats.update({'vgg_loss': vgg_loss})
            loss += vgg_loss


        scalar_stats.update({'loss': loss})

        image_stats = {'select_coord':batch['select_coord'],
                       'mask_at_box': batch['mask_at_box'],
                       'rgb_map': ret['rgb_map'],
                       'rgb_gt': batch['img_2d'].reshape(1,-1,3),
                       }
        if cfg.hard_rgb_decay_ep > -1:
            image_stats['rgb_map_hard'] = ret['rgb_map_hard']
        if 'rgb_map_val' in ret:
            image_stats['rgb_map_val'] = ret['rgb_map_val']
            image_stats['basis_rgb'] = ret['basis_rgb']
        if not cfg.nr_only:
            image_stats.update({'acc_map': ret['acc_map'],
                                })
        if cfg.neural_rendering:
            image_stats['mask_at_box'] = batch['mask_at_box'][0]
        return ret, loss, scalar_stats, image_stats


