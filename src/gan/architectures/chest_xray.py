import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import numpy as np


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity='leaky_relu')
        except Exception:
            pass
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)


def conv_out_size_same(size, stride):
    return int(np.ceil(float(size) / float(stride)))


def compute_padding_same(in_size, out_size, kernel, stride):
    res = (in_size - 1) * stride - out_size + kernel
    out_pad = 1 if (res % 2) else 0
    pad = (res + out_pad) // 2
    return pad, out_pad


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=4, stride=2, padding=1, use_sn=True, use_batch_norm=True, **kwargs):
        super().__init__()
        conv = nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False)
        if use_sn:
            conv = spectral_norm(conv)
        layers = [conv]
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Generator(nn.Module):
    def __init__(self, image_size, z_dim=100, filter_dim=64, n_blocks=3, **kwargs):
        """
        image_size:   (C, H, W)
        z_dim:        latent dimension
        filter_dim:   base channel count
        n_blocks:     desired upsampling stages
        """
        super().__init__()
        C, H, W = image_size

        # Dynamically clamp n_blocks so initial feature map >=4x4
        max_blocks = max(int(math.log2(min(H, W))) - 2, 1)
        self.n_blocks = min(n_blocks, max_blocks)
        self.filter_dim = filter_dim
        self.z_dim = z_dim

        # Compute initial dims
        self.init_h = H // (2 ** self.n_blocks)
        self.init_w = W // (2 ** self.n_blocks)
        self.init_ch = filter_dim * (2 ** self.n_blocks)

        # Project latent to feature map
        self.project = nn.Sequential(
            nn.Linear(z_dim, self.init_ch * self.init_h * self.init_w, bias=False),
            nn.BatchNorm1d(self.init_ch * self.init_h * self.init_w),
            nn.ReLU(True)
        )

        # Build upsample blocks
        blocks = []
        for i in range(self.n_blocks):
            in_ch = filter_dim * (2 ** (self.n_blocks - i))
            out_ch = filter_dim * (2 ** (self.n_blocks - i - 1))
            blocks.append(UpBlock(in_ch, out_ch))
        self.up_blocks = nn.Sequential(*blocks)

        # Final conv to image
        self.final_conv = nn.Sequential(
            nn.Conv2d(filter_dim, C, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

        self.apply(weights_init)

    def forward(self, z):
        x = self.project(z)
        x = x.view(-1, self.init_ch, self.init_h, self.init_w)
        x = self.up_blocks(x)
        return self.final_conv(x)


class Discriminator(nn.Module):
    def __init__(self, image_size, filter_dim=64, n_blocks=3, use_sn=True, use_batch_norm=True, is_critic=False, **kwargs):
        """
        image_size:      (C, H, W)
        filter_dim:      base channel count
        n_blocks:        desired downsampling stages
        """
        super().__init__()
        C, H, W = image_size

        # Clamp n_blocks similarly
        max_blocks = max(int(math.log2(min(H, W))) - 2, 1)
        self.n_blocks = min(n_blocks, max_blocks)
        layers = []
        in_ch = C
        for i in range(self.n_blocks):
            out_ch = filter_dim * (2 ** i)
            layers.append(
                DownBlock(in_ch, out_ch, use_sn=use_sn, use_batch_norm=use_batch_norm)
            )
            in_ch = out_ch
        self.features = nn.Sequential(*layers)

        # Final conv
        conv_final = nn.Conv2d(in_ch, 1, kernel_size=4, stride=1, padding=0, bias=False)
        if use_sn:
            conv_final = spectral_norm(conv_final)
        classifier = [conv_final]
        if not is_critic:
            classifier.append(nn.Sigmoid())
        self.classifier = nn.Sequential(*classifier)

        self.apply(weights_init)

    def forward(self, x):
        feat = self.features(x)
        out = self.classifier(feat)
        return out.view(-1)
