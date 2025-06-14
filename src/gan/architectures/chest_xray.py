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


def conv_out_size_same(size, stride):
    return int(np.ceil(float(size) / float(stride)))


def compute_padding_same(in_size, out_size, kernel, stride):
    # 2 * padding - out_padding = (in_size-1)*stride - out_size + kernel
    res = (in_size - 1) * stride - out_size + kernel
    out_padding = 0 if (res % 2 == 0) else 1
    padding = (res + out_padding) / 2
    return int(padding), int(out_padding)


class ConvTranspose2dSame(nn.Module):
    def __init__(self, in_ch, out_ch, in_size, out_size, kernel, stride, bias=True):
        super().__init__()
        in_h, in_w = in_size
        out_h, out_w = out_size
        pad_h, out_pad_h = compute_padding_same(in_h, out_h, kernel, stride)
        pad_w, out_pad_w = compute_padding_same(in_w, out_w, kernel, stride)
        self.conv_t = nn.ConvTranspose2d(in_ch, out_ch, kernel, stride,
                                         (pad_h, pad_w), (out_pad_h, out_pad_w), bias=bias)

    def forward(self, x):
        return self.conv_t(x)


class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.f = spectral_norm(nn.Conv2d(channels, channels//8, 1))
        self.g = spectral_norm(nn.Conv2d(channels, channels//8, 1))
        self.h = spectral_norm(nn.Conv2d(channels, channels//2, 1))
        self.v = spectral_norm(nn.Conv2d(channels//2, channels, 1))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b,c,h,w = x.size()
        f = self.f(x).view(b, -1, h*w)
        g = self.g(x).view(b, -1, h*w)
        beta = F.softmax(torch.bmm(f.permute(0,2,1), g), dim=-1)
        h_ = self.h(x).view(b, -1, h*w)
        o = torch.bmm(h_, beta).view(b, c//2, h, w)
        o = self.v(o)
        return x + self.gamma * o


class Generator(nn.Module):
    def __init__(self, image_size=(1,128,128), z_dim=100, n_blocks=4, base_ch=64):
        super().__init__()
        C,H,W = image_size
        init_spatial = H // (2**n_blocks)
        dims = [base_ch * (2**i) for i in range(n_blocks+1)]
        gen_dims = list(reversed(dims))
        # project z → feature map
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
            if curr == 32:
                seq.append(SelfAttention(in_c))
            seq += [
                nn.BatchNorm2d(in_c),
                nn.ReLU(True),
                nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False),
            ]
            blocks.append(nn.Sequential(*seq))
            curr *= 2
        self.blocks = nn.Sequential(*blocks)
        self.to_gray = nn.Sequential(nn.Conv2d(gen_dims[-1], 1, 3,1,1), nn.Tanh())
        self.apply(weights_init)

    def forward(self, z):
        x = z.view(-1, z.size(1), 1, 1)
        x = self.project(x)
        x = self.blocks(x)
        return self.to_gray(x)


class Discriminator(nn.Module):
    def __init__(self, image_size=(1,128,128), base_ch=64, use_bn=False, is_critic=True):
        super().__init__()
        C,H,W = image_size
        n_blocks = 4
        dims = [base_ch * (2**i) for i in range(n_blocks+1)]
        blocks = []
        curr = H
        in_c = 1
        for i, out_c in enumerate(dims):
            seq = [spectral_norm(nn.Conv2d(in_c, out_c, 4,2,1, bias=(i==0)))]
            if use_bn and i>0:
                seq.append(nn.BatchNorm2d(out_c))
            seq.append(nn.LeakyReLU(0.2, inplace=True))
            if curr == 32:
                seq.append(SelfAttention(out_c))
            blocks.append(nn.Sequential(*seq))
            in_c = out_c
            curr //= 2
        self.blocks = nn.Sequential(*blocks)
        self.final = spectral_norm(nn.Conv2d(dims[-1], 1, (curr, curr), 1, 0, bias=False))
        self.flatten = nn.Flatten()
        self.apply(weights_init)

    def forward(self, x):
        h = x
        for b in self.blocks:
            h = b(h)
        h = self.final(h)
        return self.flatten(h).view(-1)
