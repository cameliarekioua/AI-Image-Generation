import torch
import torch.nn as nn
from config import config



img_size = config["latent_img_size"]
img_channels = config["img_channels"]
embd_dim = config["embd_dim"]



class DownBlock(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)



class UpBlock(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)



class VAE(nn.Module):

    def __init__(self):
        super().__init__()
        down_dims = [img_channels, 32, 64, 128]
        up_dims = [4, 128, 64, 32]
        self.encoder_block = nn.Sequential(*[DownBlock(down_dims[i], down_dims[i+1]) for i in range(len(down_dims)-1)])
        self.encoder_last_layer = nn.Conv2d(down_dims[-1], 4*2, kernel_size=3, padding=1)
        self.decoder_block = nn.Sequential(*[UpBlock(up_dims[i], up_dims[i+1]) for i in range(len(down_dims)-1)])
        self.decoder_last_layer = nn.Conv2d(up_dims[-1], img_channels, kernel_size=3, padding=1)

    def encoder(self, x):
        x = self.encoder_block(x)
        mu_and_log_var = self.encoder_last_layer(x)
        mu, log_var = mu_and_log_var[:, :4, :, :], mu_and_log_var[:, 4:, :, :]
        return mu, log_var
    
    def decoder(self, z):
        x = self.decoder_block(z)
        out = self.decoder_last_layer(x)
        return out

    def forward(self, x):
        mu, log_var = self.encoder(x)    # on prédit log(sigma**2) au lieu de sigma pour des questions de stabilité numérique
        sigma = torch.exp(1/2 * log_var)
        noise = torch.randn_like(mu)
        z = mu + noise * sigma
        out = self.decoder(z)
        return out, mu, log_var
