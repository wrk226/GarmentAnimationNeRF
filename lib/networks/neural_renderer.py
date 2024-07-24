import torch.nn as nn
import torch
from math import log2
from kornia.filters import filter2d
import torch.nn.functional as F
class Blur(nn.Module):
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer('f', f)

    def forward(self, x):
        f = self.f
        f = f[None, None, :] * f[None, :, None]
        return filter2d(x, f, normalized=True)

class NeuralRenderer_palette(nn.Module):
    ''' Neural renderer class

    Args:
        n_feat (int): number of features
        input_dim (int): input dimension; if not equal to n_feat,
            it is projected to n_feat with a 1x1 convolution
        out_dim (int): output dimension
        final_actvn (bool): whether to apply a final activation (sigmoid)
        min_feat (int): minimum features
        img_size (int): output image size
        use_rgb_skip (bool): whether to use RGB skip connections
        upsample_feat (str): upsampling type for feature upsampling
        upsample_rgb (str): upsampling type for rgb upsampling
        use_norm (bool): whether to use normalization
    '''

    def __init__(
            self, n_feat=128, input_dim=128, out_dim=4, final_actvn=True,
            min_feat=32, img_size=64, use_rgb_skip=True,
            upsample_feat="nn", upsample_rgb="bilinear", use_norm=False, upsample=True, nr_upsample_steps=2,network_type='ori',
            **kwargs):
        super().__init__()
        self.upsample = upsample
        self.final_actvn = final_actvn
        self.input_dim = input_dim

        self.use_rgb_skip = use_rgb_skip
        self.use_norm = use_norm
        self.nr_upsample_steps = nr_upsample_steps
        n_blocks = int(log2(img_size) - 5)

        assert(upsample_feat in ("nn", "bilinear"))
        if upsample_feat == "nn":
            self.upsample_2 = nn.Upsample(scale_factor=2.)
        elif upsample_feat == "bilinear":
            self.upsample_2 = nn.Sequential(nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=False), Blur())

        assert(upsample_rgb in ("nn", "bilinear"))
        if upsample_rgb == "nn":
            self.upsample_rgb = nn.Upsample(scale_factor=2.)
        elif upsample_rgb == "bilinear":
            self.upsample_rgb = nn.Sequential(nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=False), Blur())

        if n_feat == input_dim:
            self.conv_in = lambda x: x
        else:
            self.conv_in = nn.Conv2d(input_dim, n_feat, 1, 1, 0)

        self.conv_layers = nn.ModuleList(
            [nn.Conv2d(n_feat, n_feat // 2, 3, 1, 1)] +
            [nn.Conv2d(max(n_feat // (2 ** (i + 1)), min_feat),
                       max(n_feat // (2 ** (i + 2)), min_feat), 3, 1, 1)
                for i in range(0, n_blocks - 1)]
        )
        if use_rgb_skip:
            self.conv_rgb = nn.ModuleList(
                [nn.Conv2d(input_dim, out_dim, 3, 1, 1)] +
                [nn.Conv2d(max(n_feat // (2 ** (i + 1)), min_feat),
                           out_dim, 3, 1, 1) for i in range(0, n_blocks)]
            )
        else:
            self.conv_rgb = nn.Conv2d(
                max(n_feat // (2 ** (n_blocks)), min_feat), 3, 1, 1)

        if use_norm:
            self.norms = nn.ModuleList([
                nn.InstanceNorm2d(max(n_feat // (2 ** (i + 1)), min_feat))
                for i in range(n_blocks)
            ])
        self.actvn = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        if self.nr_upsample_steps>0:
            self.upsample = True

        net = self.conv_in(x)

        if self.use_rgb_skip:
            if self.upsample:
                rgb = self.upsample_rgb(self.conv_rgb[0](x))
            else:
                rgb = self.conv_rgb[0](x)

        for idx, layer in enumerate(self.conv_layers):

            if self.upsample:
                hid = layer(self.upsample_2(net))
            else:
                hid = layer(net)
            if self.use_norm:
                hid = self.norms[idx](hid)
            net = self.actvn(hid)

            if self.use_rgb_skip:
                rgb = rgb + self.conv_rgb[idx + 1](net)
                if idx < len(self.conv_layers) - 1:
                    if idx > self.nr_upsample_steps-2:
                        self.upsample = False
                    if self.upsample:
                        rgb = self.upsample_rgb(rgb)
                    else:
                        rgb = rgb

        if not self.use_rgb_skip:
            rgb = self.conv_rgb(net)

        if self.final_actvn:
            rgb = torch.sigmoid(rgb)
        return rgb

class NeuralRenderer(nn.Module):
    ''' Neural renderer class

    Args:
        n_feat (int): number of features
        input_dim (int): input dimension; if not equal to n_feat,
            it is projected to n_feat with a 1x1 convolution
        out_dim (int): output dimension
        final_actvn (bool): whether to apply a final activation (sigmoid)
        min_feat (int): minimum features
        img_size (int): output image size
        use_rgb_skip (bool): whether to use RGB skip connections
        upsample_feat (str): upsampling type for feature upsampling
        upsample_rgb (str): upsampling type for rgb upsampling
        use_norm (bool): whether to use normalization
    '''

    def __init__(
            self, n_feat=128, input_dim=128, out_dim=4, final_actvn=True,
            min_feat=32, img_size=64, use_rgb_skip=True,
            upsample_feat="nn", upsample_rgb="bilinear", use_norm=False, upsample=True, nr_upsample_steps=2,network_type='ori',
            **kwargs):
        super().__init__()
        self.upsample = upsample
        self.final_actvn = final_actvn
        self.input_dim = input_dim

        self.use_rgb_skip = use_rgb_skip
        self.use_norm = use_norm
        self.nr_upsample_steps = nr_upsample_steps
        n_blocks = int(log2(img_size) - 5)

        assert(upsample_feat in ("nn", "bilinear"))
        if upsample_feat == "nn":
            self.upsample_2 = nn.Upsample(scale_factor=2.)
        elif upsample_feat == "bilinear":
            self.upsample_2 = nn.Sequential(nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=False), Blur())

        assert(upsample_rgb in ("nn", "bilinear"))
        if upsample_rgb == "nn":
            self.upsample_rgb = nn.Upsample(scale_factor=2.)
        elif upsample_rgb == "bilinear":
            self.upsample_rgb = nn.Sequential(nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=False), Blur())

        if n_feat == input_dim:
            self.conv_in = lambda x: x
        else:
            self.conv_in = nn.Conv2d(input_dim, n_feat, 1, 1, 0)

        self.conv_layers = nn.ModuleList(
            [nn.Conv2d(n_feat, n_feat // 2, 3, 1, 1)] +
            [nn.Conv2d(max(n_feat // (2 ** (i + 1)), min_feat),
                       max(n_feat // (2 ** (i + 2)), min_feat), 3, 1, 1)
                for i in range(0, n_blocks - 1)]
        )
        if use_rgb_skip:
            self.conv_rgb = nn.ModuleList(
                [nn.Conv2d(input_dim, out_dim, 3, 1, 1)] +
                [nn.Conv2d(max(n_feat // (2 ** (i + 1)), min_feat),
                           out_dim, 3, 1, 1) for i in range(0, n_blocks)]
            )
        else:
            self.conv_rgb = nn.Conv2d(
                max(n_feat // (2 ** (n_blocks)), min_feat), 3, 1, 1)

        if use_norm:
            self.norms = nn.ModuleList([
                nn.InstanceNorm2d(max(n_feat // (2 ** (i + 1)), min_feat))
                for i in range(n_blocks)
            ])
        self.actvn = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        if self.nr_upsample_steps>0:
            self.upsample = True

        net = self.conv_in(x)

        if self.use_rgb_skip:
            if self.upsample:
                rgb = self.upsample_rgb(self.conv_rgb[0](x))
            else:
                rgb = self.conv_rgb[0](x)

        for idx, layer in enumerate(self.conv_layers):

            if self.upsample:
                hid = layer(self.upsample_2(net))
            else:
                hid = layer(net)
            if self.use_norm:
                hid = self.norms[idx](hid)
            net = self.actvn(hid)

            if self.use_rgb_skip:
                rgb = rgb + self.conv_rgb[idx + 1](net)
                if idx < len(self.conv_layers) - 1:
                    if idx > self.nr_upsample_steps-2:
                        self.upsample = False
                    if self.upsample:
                        rgb = self.upsample_rgb(rgb)
                    else:
                        rgb = rgb

        if not self.use_rgb_skip:
            rgb = self.conv_rgb(net)

        if self.final_actvn:
            rgb = torch.sigmoid(rgb)
        return rgb

class NeuralRenderer_multiscale(nn.Module):
    def __init__(self, img_size=256, upsample=False,nr_upsample_steps=2,input_dim=128):
        super().__init__()
        self.nr_128 = NeuralRenderer(img_size=img_size, upsample=upsample,nr_upsample_steps=nr_upsample_steps,input_dim=input_dim, final_actvn=False)
        self.nr_64 = NeuralRenderer(img_size=img_size, upsample=upsample,nr_upsample_steps=nr_upsample_steps+1,input_dim=input_dim, final_actvn=False)
        self.nr_32 = NeuralRenderer(img_size=img_size, upsample=upsample,nr_upsample_steps=nr_upsample_steps+2,input_dim=input_dim, final_actvn=False)

    def forward(self, x):
        feat_map = x
        out_1 = self.nr_128(feat_map)
        feat_map_64 = F.avg_pool2d(feat_map, kernel_size=2, stride=2)
        out_2 = self.nr_64(feat_map_64)
        feat_map_32 = F.avg_pool2d(feat_map_64, kernel_size=2, stride=2)
        out_3 = self.nr_32(feat_map_32)
        out = torch.sigmoid(torch.sum(torch.stack([out_1,out_2,out_3]),dim=0))
        return out



# main
if __name__ == '__main__':
    # nr = NeuralRenderer(img_size=256)
    nr_128 = NeuralRenderer(img_size=512, upsample=False,nr_upsample_steps=2,input_dim=64)
    nr_64 = NeuralRenderer(img_size=512, upsample=False,nr_upsample_steps=3,input_dim=64)
    nr_32 = NeuralRenderer(img_size=512, upsample=False,nr_upsample_steps=4,input_dim=64)

    # nr = nn.Sequential(
    #                     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
    #                     nn.Conv2d(64, 64, 3, 1, 1),
    #                    nn.LeakyReLU(0.2, inplace=True),
    #                     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
    #                    nn.Conv2d(64, 32, 3, 1, 1),
    #                    nn.LeakyReLU(0.2, inplace=True),
    #
    #                    nn.Conv2d(32, 32, 3, 1, 1),
    #                    nn.LeakyReLU(0.2, inplace=True),
    #
    #                    nn.Conv2d(32, 32, 3, 1, 1),
    #                    nn.LeakyReLU(0.2, inplace=True),
    #
    #                    nn.Conv2d(32, 3, 3, 1, 1),
    #                    nn.Sigmoid())

    feat_map = torch.randn(1, 64, 128,128)
    out_1 = nr_128(feat_map)
    feat_map_64 = F.avg_pool2d(feat_map, kernel_size=2, stride=2)
    out_2 = nr_64(feat_map_64)
    feat_map_32 = F.avg_pool2d(feat_map_64, kernel_size=2, stride=2)
    out_3 = nr_32(feat_map_32)
    print(out_1.shape)