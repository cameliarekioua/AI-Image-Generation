import torch.nn as nn
from config import config

img_size = config["img_size"]
img_channels = config["img_channels"]

embd_dim = 32


class ConvBlock(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class ImgToEmbdNetwork(nn.Module):

    def __init__(self, channels=[3, 8, 16, 32, 64, 128]):
        super().__init__()
        self.blocks = nn.Sequential(*[ConvBlock(channels[i], channels[i+1]) for i in range(len(channels)-1)])
        self.proj = nn.Linear(channels[-1]*8*8, embd_dim)
    
    def forward(self, x):
        x = self.blocks(x)
        x = x.view(x.shape[0], -1)
        x = self.proj(x)
        return x


class TextToEmbdNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(77, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )
        self.proj = nn.Linear(128, embd_dim)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.proj(x)
        return x

