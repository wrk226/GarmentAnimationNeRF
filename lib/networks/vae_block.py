import torch
from torch import nn
from torch.nn import functional as F
class PrintLayer(nn.Module):
    def __init__(self, idx=None):
        super(PrintLayer, self).__init__()
        self.idx = idx

    def forward(self, x):
        # Do your print / debug stuff here
        # print(x.shape, self.idx )
        return x

class Single_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.single_conv = nn.Sequential(
                                        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                                        nn.InstanceNorm2d(out_channels),
                                        nn.LeakyReLU(inplace=True),
                                    )

    def forward(self, x):
        return self.single_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.single_conv = nn.Sequential(
                                        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                                        nn.InstanceNorm2d(out_channels),
                                        nn.LeakyReLU(inplace=True),
                                    )

    def forward(self, x):
        return self.single_conv(x)



class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = Single_conv(in_channels, out_channels)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        if x2 is not None:
            # diffY = x2.shape[2] - x1.size()[2]
            # diffX = x2.shape[3] - x1.size()[3]
            # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
            #                 diffY // 2, diffY - diffY // 2])
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        res = self.conv(x)
        return res



class AutoEncoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 latent_dim: int = 512,
                 ae_type = "ae",
                 skip_connection = True,
                 use_flatten = True,
                 n_down=6) -> None:
        super(AutoEncoder, self).__init__()
        self.use_flatten = use_flatten
        self.n_down=n_down
        flatten_dim = 1024
        self.norm_method = nn.InstanceNorm2d
        self.skip_connection = skip_connection
        self.latent_dim = latent_dim
        self.ae_type = ae_type

        # Build Encoder
        self.encoder_input = Single_conv(in_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        if n_down == 6:
            self.down5 = Down(512, 1024)
            self.down6 = Down(1024, flatten_dim)
        else:
            flatten_dim = 512

        if ae_type == "ae":
            self.fc = nn.Linear(flatten_dim*4, latent_dim)
        elif ae_type == "vae":
            self.fc_mu = nn.Linear(flatten_dim*4, latent_dim)
            self.fc_var = nn.Linear(flatten_dim*4, latent_dim)
        else:
            raise ValueError("vae_type must be either 'ae' or 'vae'")


        # Build Decoder
        # boolean to int
        skip_connection_flag = skip_connection * 1
        if n_down == 6:
            self.decoder_input = nn.Linear(latent_dim, 1024 * 4)
            self.decoder_input_dim = 1024
            self.up1 = Up(1024+1024*skip_connection_flag, 1024)
            self.up2 = Up(1024+512*skip_connection_flag, 512)
        else:
            self.decoder_input = nn.Linear(latent_dim, 512 * 4)
            self.decoder_input_dim = 512
        self.up3 = Up(512+256*skip_connection_flag, 256)
        self.up4 = Up(256+128*skip_connection_flag, 128)
        if out_channels <= 32:
            self.up5 = Up(128+64*skip_connection_flag, 64)
            self.up6 = Up(64+32*skip_connection_flag, 32)
            self.decoder_output = nn.Sequential(
                                        nn.Sequential(
                                        nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
                                        nn.Tanh()
                                        )
                                    )
        elif out_channels == 64:
            self.up5 = Up(128+64*skip_connection_flag, 64)
            self.up6 = Up(64+32*skip_connection_flag, 64)
            self.decoder_output = nn.Sequential(
                                        nn.Sequential(
                                        nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
                                        nn.Tanh()
                                        )
                                    )
        elif out_channels == 128:
            self.up5 = Up(128+64*skip_connection_flag, 128)
            self.up6 = Up(128+32*skip_connection_flag, 128)
            self.decoder_output = nn.Sequential(
                                        nn.Sequential(
                                        nn.Conv2d(128, out_channels, kernel_size=3, padding=1),
                                        nn.Tanh()
                                        )
                                    )

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def encode(self, input):
        # encode
        x1 = self.encoder_input(input)  # [1,3,128,128] -> [1,32,128,128]
        x2 = self.down1(x1)     # [1,32,128,128] -> [1,64,64,64]
        x3 = self.down2(x2)     # [1,64,64,64] -> [1,128,32,32]
        x4 = self.down3(x3)     # [1,128,32,32] -> [1,256,16,16]
        feat = self.down4(x4)     # [1,256,16,16] -> [1,512,8,8]
        return feat

    def decode(self, feat):
        result = self.up3(feat) # [1,512,8,8] -> [1,256,16,16]
        result = self.up4(result) # [1,256,16,16] -> [1,128,32,32]
        result = self.up5(result) # [1,128,32,32] -> [1,64,64,64]
        result = self.up6(result) # [1,64,64,64] -> [1,32,128,128]
        output = self.decoder_output(result) # [1,32,128,128] -> [1,output,128,128]
        return output

    def forward(self, input):
        # encode
        x1 = self.encoder_input(input)  # [1,3,128,128] -> [1,32,128,128]
        x2 = self.down1(x1)     # [1,32,128,128] -> [1,64,64,64]
        x3 = self.down2(x2)     # [1,64,64,64] -> [1,128,32,32]
        x4 = self.down3(x3)     # [1,128,32,32] -> [1,256,16,16]
        x5 = self.down4(x4)     # [1,256,16,16] -> [1,512,8,8]
        if self.n_down == 6:
            x6 = self.down5(x5)     # [1,512,8,8] -> [1,1024,4,4]
            result = self.down6(x6)     # [1,1024,4,4] -> [1,2048,2,2]
        else:
            result = x5

        # reparameterize
        if self.ae_type == "ae":
            if self.use_flatten:
                result = torch.flatten(result, start_dim=1)
                z = self.fc(result)
        else:
            raise ValueError("vae_type must be either 'ae' or 'vae'")


        # decode
        if self.use_flatten:
            result = self.decoder_input(z) # [1,512] -> [1,1024*4]
            result = result.view(-1, self.decoder_input_dim, 2, 2) # [1,1024*4] -> [1,1024,2,2]

        if self.n_down == 6:
            result = self.up1(result, x6 if self.skip_connection else None) # [1,2048,2,2] -> [1,1024,4,4]
            result = self.up2(result, x5 if self.skip_connection else None) # [1,1024,4,4] -> [1,512,8,8]
        result = self.up3(result, x4 if self.skip_connection else None) # [1,512,8,8] -> [1,256,16,16]
        result = self.up4(result, x3 if self.skip_connection else None) # [1,256,16,16] -> [1,128,32,32]
        result = self.up5(result, x2 if self.skip_connection else None) # [1,128,32,32] -> [1,64,64,64]
        result = self.up6(result, x1 if self.skip_connection else None) # [1,64,64,64] -> [1,32,128,128]
        output = self.decoder_output(result) # [1,32,128,128] -> [1,output,128,128]
        return output

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}

    def sample(self,
               num_samples:int,
               current_device: int, ):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

# main
if __name__ == '__main__':
    model = AutoEncoder(3, 128, skip_connection=True)
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))
    # print(model)
    test_input = torch.randn(1, 3, 128, 128)
    test_out = model(test_input)
    print(test_out.shape)
