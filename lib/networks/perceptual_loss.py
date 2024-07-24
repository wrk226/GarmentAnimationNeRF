import torch.nn as nn
import torch
import torchvision
from torchvision import models
from collections import namedtuple
from lib.config import cfg

class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, weights=None):
        super(vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(weights=weights).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)

        return out


class PNetLin(nn.Module):
    def __init__(self, pnet_type='vgg', pnet_tune=False, use_dropout=False, spatial=False, device='cuda'):
        super(PNetLin, self).__init__()

        self.pnet_type = pnet_type
        self.pnet_tune = pnet_tune
        self.spatial = spatial

        self.chns = [64,128,256,512,512]

        self.net = vgg16(requires_grad=False, weights=torchvision.models.vgg.VGG16_Weights.IMAGENET1K_V1)

        self.net.to(device)


    def normalize_tensor(self, in_feat,eps=1e-10):
        # norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1)).view(in_feat.size()[0],1,in_feat.size()[2],in_feat.size()[3]).repeat(1,in_feat.size()[1],1,1)
        norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1)).view(in_feat.size()[0],1,in_feat.size()[2],in_feat.size()[3])
        return in_feat/(norm_factor.expand_as(in_feat)+eps)

    def forward(self, in0, in1):

        outs0 = self.net.forward(in0)
        outs1 = self.net.forward(in1)

        feats0 = {}
        feats1 = {}
        diffs = [0]*len(outs0)

        # todo : comment this
        for (kk,out0) in enumerate(outs0):

            # use l2 distance
            if cfg.vgg_type == 'l2':
                feats0[kk] = self.normalize_tensor(outs0[kk])
                feats1[kk] = self.normalize_tensor(outs1[kk])
                diffs[kk] = (feats0[kk]-feats1[kk])**2
            elif cfg.vgg_type == 'normcos':
                # use consine distance
                feats0[kk] = self.normalize_tensor(outs0[kk])
                feats1[kk] = self.normalize_tensor(outs1[kk])
                diffs[kk] = 1. - torch.sum(feats0[kk]*feats1[kk],dim=1,keepdim=True)
            elif cfg.vgg_type == 'cos':
                # Step 1: Calculate the dot product
                dot_product = torch.sum(outs0[kk] * outs1[kk], dim=1, keepdim=True)

                # Step 2: Calculate the L2 norms
                norm_outs0 = torch.sqrt(torch.sum(outs0[kk]**2, dim=1, keepdim=True) + 1e-10)  # added epsilon for numerical stability
                norm_outs1 = torch.sqrt(torch.sum(outs1[kk]**2, dim=1, keepdim=True) + 1e-10)

                # Step 3: Compute the cosine similarity
                cosine_similarity = dot_product / (norm_outs0 * norm_outs1)

                # Convert similarity to distance
                diffs[kk] = 1. - cosine_similarity

        val = torch.mean(torch.flatten(diffs[0]))
        val = val + torch.mean(torch.flatten(diffs[1]))
        val = val + torch.mean(torch.flatten(diffs[2]))
        val = val + torch.mean(torch.flatten(diffs[3]))
        val = val + torch.mean(torch.flatten(diffs[4]))

        return val
if __name__ == '__main__':
    vgg_loss = PNetLin()
    import cv2
    # load image from seen_pose_epoch0085_res256_frame0021_view0000.png
    # gt = cv2.imread('seen_pose_epoch0320_res256_frame0021_view0000_gt.png')
    # pred1 = cv2.imread('seen_pose_epoch0001_res256_frame0021_view0000.png')
    # pred2 = cv2.imread('seen_pose_epoch0013_res256_frame0021_view0000.png')
    # pred3 = cv2.imread('seen_pose_epoch0085_res256_frame0021_view0000.png')
    # pred4 = cv2.imread('seen_pose_epoch0088_res256_frame0021_view0000.png')
    # pred5 = cv2.imread('seen_pose_epoch0320_res256_frame0021_view0000.png')
    # gt = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).float().cuda()
    # pred1 = torch.from_numpy(pred1).permute(2, 0, 1).unsqueeze(0).float().cuda()
    # pred2 = torch.from_numpy(pred2).permute(2, 0, 1).unsqueeze(0).float().cuda()
    # pred3 = torch.from_numpy(pred3).permute(2, 0, 1).unsqueeze(0).float().cuda()
    # pred4 = torch.from_numpy(pred4).permute(2, 0, 1).unsqueeze(0).float().cuda()
    # pred5 = torch.from_numpy(pred5).permute(2, 0, 1).unsqueeze(0).float().cuda()
    #
    # output = vgg_loss(gt, pred1).squeeze()
    # output1 = vgg_loss(gt, pred2).squeeze()
    # output2 = vgg_loss(gt, pred3).squeeze()
    # output3 = vgg_loss(gt, pred4).squeeze()
    # output4 = vgg_loss(gt, pred5).squeeze()
    # print(output.item(), output1.item(), output2.item(), output3.item(), output4.item())
    gt = torch.ones(1, 3, 16, 16).cuda()
    pred = torch.rand(1, 3, 16, 16).cuda()
    output = vgg_loss(gt, pred).squeeze()
    print(output.item())
