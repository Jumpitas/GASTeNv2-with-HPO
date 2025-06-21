import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


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
        self.f = spectral_norm(nn.Conv2d(channels, channels // 8, 1))
        self.g = spectral_norm(nn.Conv2d(channels, channels // 8, 1))
        self.h = spectral_norm(nn.Conv2d(channels, channels // 2, 1))
        self.v = spectral_norm(nn.Conv2d(channels // 2, channels, 1))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, h, w = x.size()
        f = self.f(x).view(b, -1, h*w)                  # [B, C/8, HW]
        g = self.g(x).view(b, -1, h*w)                  # [B, C/8, HW]
        beta = F.softmax(torch.bmm(f.permute(0,2,1), g), dim=-1)  # [B, HW, HW]
        h_ = self.h(x).view(b, -1, h*w)                 # [B, C/2, HW]
        o = torch.bmm(h_, beta).view(b, c//2, h, w)     # [B, C/2, H, W]
        o = self.v(o)                                   # [B, C, H, W]
        return x + self.gamma * o


class ResidualBlockUp(nn.Module):
    """ ResBlock with nearest-neighbor upsampling + two 3×3 convs """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv1 = spectral_norm(nn.Conv2d(in_ch, out_ch, 3, 1, 1))
        self.conv2 = spectral_norm(nn.Conv2d(out_ch, out_ch, 3, 1, 1))
        self.skip  = spectral_norm(nn.Conv2d(in_ch, out_ch, 1, 1, 0))
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        x0 = x
        x  = F.relu(self.bn1(x))
        x  = self.upsample(x)
        x  = self.conv1(x)
        x  = F.relu(self.bn2(x))
        x  = self.conv2(x)
        x0 = self.skip(self.upsample(x0))
        return x + x0


class ResidualBlockDown(nn.Module):
    """ ResBlock with two 3×3 convs + average-pool downsampling """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = spectral_norm(nn.Conv2d(in_ch, in_ch, 3, 1, 1))
        self.conv2 = spectral_norm(nn.Conv2d(in_ch, out_ch, 3, 1, 1))
        self.skip  = spectral_norm(nn.Conv2d(in_ch, out_ch, 1, 1, 0))
        self.downsample = nn.AvgPool2d(2)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.bn2 = nn.BatchNorm2d(in_ch)

    def forward(self, x):
        x0 = x
        x  = F.relu(self.bn1(x))
        x  = self.conv1(x)
        x  = F.relu(self.bn2(x))
        x  = self.conv2(x)
        x  = self.downsample(x)
        x0 = self.skip(self.downsample(x0))
        return x + x0


class Generator(nn.Module):
    def __init__(self, z_dim=128, base_ch=64, img_ch=3, n_blocks=4):
        """
        z_dim:    latent size
        base_ch:  how many filters to start with (will be doubled each block)
        img_ch:   output channels (3 for RGB)
        n_blocks: number of up-sampling blocks (2**n_blocks must match final H/W)
        """
        super().__init__()

        # compute initial spatial size
        final_size = 96  # e.g. for STL-10
        init_spatial = final_size // (2 ** n_blocks)  # must be integer
        # channels per level: base_ch * 2**i
        dims = [base_ch * (2 ** i) for i in reversed(range(n_blocks+1))]
        # dims[0] is the coarsest, dims[-1] the finest

        # project z → feature map
        self.project = nn.Sequential(
            nn.Linear(z_dim, dims[0] * init_spatial * init_spatial),
            nn.ReLU(True),
            nn.BatchNorm1d(dims[0] * init_spatial * init_spatial),
        )

        # build up-sampling resblocks
        self.blocks = nn.ModuleList()
        curr_size = init_spatial
        for i in range(n_blocks):
            in_ch  = dims[i]
            out_ch = dims[i+1]
            self.blocks.append(ResidualBlockUp(in_ch, out_ch))
            curr_size *= 2
            # insert attention at 24×24 and 48×48
            if curr_size in (24, 48):
                self.blocks.append(SelfAttention(out_ch))

        # final to RGB
        self.to_rgb = nn.Sequential(
            spectral_norm(nn.Conv2d(dims[-1], img_ch, 3, 1, 1)),
            nn.Tanh()
        )

        self.init_spatial = init_spatial
        self.dims = dims
        self.apply(weights_init)

    def forward(self, z):
        # z: [B, z_dim]
        B = z.size(0)
        x = self.project(z)
        x = x.view(B, self.dims[0], self.init_spatial, self.init_spatial)
        for blk in self.blocks:
            x = blk(x)
        return self.to_rgb(x)


class Discriminator(nn.Module):
    def __init__(self, base_ch=64, img_ch=3, use_bn=False):
        """
        base_ch: number of filters in first block (doubled each time)
        img_ch:  input channels (3 for RGB)
        use_bn:  whether to use BatchNorm after conv1
        """
        super().__init__()
        final_size = 96
        n_blocks   = 4
        dims = [base_ch * (2 ** i) for i in range(n_blocks+1)]

        # build down-sampling resblocks
        self.blocks = nn.ModuleList()
        curr_size = final_size
        in_ch = img_ch
        for i, out_ch in enumerate(dims):
            self.blocks.append(ResidualBlockDown(in_ch, out_ch))
            curr_size //= 2
            in_ch = out_ch
            # insert attention at 48×48 and 24×24 *after* downsampling
            if curr_size in (48, 24):
                self.blocks.append(SelfAttention(out_ch))

        # final 1×1 conv → scalar
        self.final = spectral_norm(nn.Conv2d(dims[-1], 1, curr_size, 1, 0))
        self.flatten = nn.Flatten()

        self.apply(weights_init)

    def forward(self, x):
        h = x
        for blk in self.blocks:
            h = blk(h)
        h = self.final(h)
        return self.flatten(h).view(-1)
