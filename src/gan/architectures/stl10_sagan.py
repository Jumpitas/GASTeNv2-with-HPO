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
        beta = F.softmax(torch.bmm(f.permute(0,2,1), g), dim=-1)
        h_ = self.h(x).view(b, -1, h*w)          # value
        o = torch.bmm(h_, beta).view(b, c//2, h, w)
        o = self.v(o)
        return x + self.gamma * o

class ResBlockUp(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.skip  = nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=False)

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
        self.act = nn.LeakyReLU(0.2, inplace=False)

    def forward(self, x):
        skip = self.skip(x)
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        return self.act(x + skip)

class Generator(nn.Module):
    def __init__(self, z_dim=128, base_ch=512, img_ch=3):
        super().__init__()
        self.z_dim   = z_dim
        self.base_ch = base_ch
        self.img_ch  = img_ch

        # ───❶ Output features for a 6×6 map instead of 4×4:
        self.fc = spectral_norm(nn.Linear(z_dim, base_ch * 6 * 6))

        # Four up‐sampling blocks: 6→12→24→48→96
        self.net = nn.Sequential(
            ResBlockUp(base_ch,       base_ch//2),   #  6×6 →  12×12
            SelfAttention(base_ch//2),
            ResBlockUp(base_ch//2,    base_ch//4),   # 12×12 →  24×24
            ResBlockUp(base_ch//4,    base_ch//8),   # 24×24 →  48×48
            SelfAttention(base_ch // 8),
            ResBlockUp(base_ch//8,    base_ch//16),  # 48×48 →  96×96
            nn.BatchNorm2d(base_ch//16),
            nn.ReLU(inplace=False),
            spectral_norm(nn.Conv2d(base_ch//16, img_ch, 3, 1, 1)),
            nn.Tanh()
        )


    def forward(self, z):
        # ───❷ Reshape into (batch, base_ch, 6, 6)
        x = self.fc(z).view(z.size(0), self.base_ch, 6, 6)
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, base_ch=64, img_ch=3):
        super().__init__()
        self.base_ch = base_ch
        self.img_ch  = img_ch

        self.net = nn.Sequential(
            spectral_norm(nn.Conv2d(img_ch, base_ch, 3, 1, 1)),   # 96×96 → 96×96
            nn.LeakyReLU(0.2, inplace=False),
            ResBlockDown(base_ch,       base_ch*2),   # 96×96 → 48×48
            SelfAttention(base_ch*2),
            ResBlockDown(base_ch*2,     base_ch*4),   # 48×48 → 24×24
            ResBlockDown(base_ch*4,     base_ch*8),   # 24×24 → 12×12
            ResBlockDown(base_ch*8,     base_ch*16),  # 12×12 →  6×6
            nn.LeakyReLU(0.2, inplace=False),
            # Final conv: 6×6 → 3×3 (kernel=4, stride=2, padding=1)
            spectral_norm(nn.Conv2d(base_ch*16, base_ch*16, 4, 2, 1)),  # 6×6 → 3×3
            nn.LeakyReLU(0.2, inplace=False),
        )
        self.final = spectral_norm(nn.Conv2d(base_ch*16, 1, 3, 1, 0))

    def forward(self, x):
        features = self.net(x)
        out = self.final(features)
        return out.view(out.size(0))
