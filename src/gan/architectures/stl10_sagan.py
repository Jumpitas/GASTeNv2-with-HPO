import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.f = spectral_norm(nn.Conv2d(in_channels, in_channels//8, 1))
        self.g = spectral_norm(nn.Conv2d(in_channels, in_channels//8, 1))
        self.h = spectral_norm(nn.Conv2d(in_channels, in_channels//2, 1))
        self.v = spectral_norm(nn.Conv2d(in_channels//2, in_channels, 1))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, h, w = x.size()
        f = self.f(x).view(b, -1, h*w)           # key
        g = self.g(x).view(b, -1, h*w)           # query
        beta = F.softmax(torch.bmm(f.permute(0,2,1), g), dim=-1)  # attention map
        h_ = self.h(x).view(b, -1, h*w)          # value
        o = torch.bmm(h_, beta).view(b, c//2, h, w)
        o = self.v(o)
        return x + self.gamma * o

class ResBlockUp(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = spectral_norm(nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1))
        self.conv2 = spectral_norm(nn.Conv2d(out_ch, out_ch, 3, 1, 1))
        self.skip  = spectral_norm(nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1))
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(True)

    def forward(self, x):
        skip = self.skip(x)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.act(x + skip)

class ResBlockDown(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = spectral_norm(nn.Conv2d(in_ch, in_ch, 3, 1, 1))
        self.conv2 = spectral_norm(nn.Conv2d(in_ch, out_ch, 4, 2, 1))
        self.skip  = spectral_norm(nn.Conv2d(in_ch, out_ch, 4, 2, 1))
        self.act = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        skip = self.skip(x)
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        return self.act(x + skip)

class Generator(nn.Module):
    def __init__(self, z_dim=128, base_ch=256, img_ch=3):
        super().__init__()
        # ←— store these so you can later do G.z_dim, G.base_ch, G.img_ch
        self.z_dim   = z_dim
        self.base_ch = base_ch
        self.img_ch  = img_ch

        self.fc = spectral_norm(nn.Linear(z_dim, base_ch*4*4))
        self.net = nn.Sequential(
            ResBlockUp(base_ch, base_ch//2),    # 8×8
            SelfAttention(base_ch//2),
            ResBlockUp(base_ch//2, base_ch//4), # 16×16
            ResBlockUp(base_ch//4, base_ch//8), # 32×32
            ResBlockUp(base_ch//8, base_ch//16),# 64×64
            nn.BatchNorm2d(base_ch//16),
            nn.ReLU(True),
            spectral_norm(nn.Conv2d(base_ch//16, img_ch, 3, 1, 1)),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z).view(z.size(0), -1, 4, 4)
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, base_ch=64, img_ch=3):
        super().__init__()
        # store as attributes if you ever need them
        self.base_ch = base_ch
        self.img_ch  = img_ch

        self.net = nn.Sequential(
            spectral_norm(nn.Conv2d(img_ch, base_ch, 3, 1, 1)), #128×128
            nn.LeakyReLU(0.2, True),
            ResBlockDown(base_ch, base_ch*2),   #64
            SelfAttention(base_ch*2),
            ResBlockDown(base_ch*2, base_ch*4), #32
            ResBlockDown(base_ch*4, base_ch*8), #16
            ResBlockDown(base_ch*8, base_ch*16),#8
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(base_ch*16, 1, 4, 1, 0)), # 5×5→1×1
            nn.Flatten()
        )

    def forward(self, x):
        return self.net(x).squeeze()
