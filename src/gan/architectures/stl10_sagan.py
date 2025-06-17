import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import numpy as np


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if getattr(m, 'bias', None) is not None:
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)


class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.f = spectral_norm(nn.Conv2d(channels, channels//8, 1))
        self.g = spectral_norm(nn.Conv2d(channels, channels//8, 1))
        self.h = spectral_norm(nn.Conv2d(channels, channels//2, 1))
        self.v = spectral_norm(nn.Conv2d(channels//2, channels, 1))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b,c,h,w = x.shape
        f = self.f(x).view(b, -1, h*w)              # [B, C//8, HW]
        g = self.g(x).view(b, -1, h*w)              # [B, C//8, HW]
        beta = F.softmax(torch.bmm(f.permute(0,2,1), g), dim=-1)  # [B, HW, HW]
        h_ = self.h(x).view(b, -1, h*w)             # [B, C//2, HW]
        o = torch.bmm(h_, beta).view(b, c//2, h, w)  # [B, C//2, H, W]
        o = self.v(o)                               # [B, C, H, W]
        return x + self.gamma * o


class Generator(nn.Module):
    def __init__(self, z_dim=128, base_ch=64, img_ch = 3, n_blocks=4):
        super().__init__()
        # STL‑10 images are 3×96×96
        init_spatial = 96 // (2**n_blocks)  # = 6 when n_blocks=4
        dims = [base_ch * (2**i) for i in range(n_blocks+1)]
        gen_dims = list(reversed(dims))  # e.g. [512,256,128,64,32]

        # project z → small feature map
        self.project = nn.Sequential(
            nn.ConvTranspose2d(z_dim, gen_dims[0], init_spatial, 1, 0, bias=False),
            nn.BatchNorm2d(gen_dims[0]),
            nn.ReLU(True),
        )

        # upsampling blocks
        blocks = []
        curr = init_spatial
        for i in range(n_blocks):
            in_c, out_c = gen_dims[i], gen_dims[i+1]
            seq = [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)]
            if curr == 24:  # insert attention at 24×24 resolution
                seq.append(SelfAttention(in_c))
            seq += [
                nn.BatchNorm2d(in_c),
                nn.ReLU(True),
                nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False),
            ]
            blocks.append(nn.Sequential(*seq))
            curr *= 2
        self.blocks = nn.Sequential(*blocks)

        # final to RGB
        self.to_rgb = nn.Sequential(
            nn.Conv2d(gen_dims[-1], 3, 3, 1, 1, bias=True),
            nn.Tanh()
        )

        self.apply(weights_init)

    def forward(self, z):
        # z: [B, z_dim]
        x = z.view(-1, z.size(1), 1, 1)
        x = self.project(x)
        x = self.blocks(x)
        return self.to_rgb(x)


class Discriminator(nn.Module):
    def __init__(self, base_ch=64, img_ch =3, use_bn=False):
        super().__init__()
        # downsample from 3×96×96 to 1×1×1
        n_blocks = 4
        dims = [base_ch * (2**i) for i in range(n_blocks+1)]  # [64,128,256,512,1024]
        blocks = []
        curr = 96
        in_c = 3
        for i, out_c in enumerate(dims):
            seq = [spectral_norm(nn.Conv2d(in_c, out_c, 4, 2, 1, bias=(i==0)))]
            if use_bn and i > 0:
                seq.append(nn.BatchNorm2d(out_c))
            seq.append(nn.LeakyReLU(0.2, inplace=True))
            if curr == 48:  # attention at 48×48
                seq.append(SelfAttention(out_c))
            blocks.append(nn.Sequential(*seq))
            in_c = out_c
            curr //= 2
        self.blocks = nn.Sequential(*blocks)

        # final 1×1 conv → scalar
        self.final = spectral_norm(nn.Conv2d(dims[-1], 1, curr, 1, 0, bias=False))
        self.flatten = nn.Flatten()

        self.apply(weights_init)

    def forward(self, x):
        h = x
        for b in self.blocks:
            h = b(h)
        h = self.final(h)
        return self.flatten(h).view(-1)
