from __future__ import annotations
import math, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch.utils.checkpoint import checkpoint


# ----------------------------------------------------------------
#  initialisation helpers
# ----------------------------------------------------------------
def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


# ----------------------------------------------------------------
#  Self-Attention (with activation‐checkpointing)
# ----------------------------------------------------------------
class SelfAttention(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.f = spectral_norm(nn.Conv2d(ch,   ch // 8, 1))
        self.g = spectral_norm(nn.Conv2d(ch,   ch // 8, 1))
        self.h = spectral_norm(nn.Conv2d(ch,   ch // 2, 1))
        self.v = spectral_norm(nn.Conv2d(ch // 2, ch,      1))
        self.gamma = nn.Parameter(torch.zeros(1))

    def _core(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        f = self.f(x).view(b, -1, h * w)
        g = self.g(x).view(b, -1, h * w)
        beta = torch.softmax(torch.bmm(f.transpose(1, 2), g), dim=-1)  # B×HW×HW
        h_ = self.h(x).view(b, -1, h * w)
        o  = torch.bmm(h_, beta).view(b, c // 2, h, w)
        return x + self.gamma * self.v(o)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return checkpoint(self._core, x)


# ----------------------------------------------------------------
#  Residual blocks
# ----------------------------------------------------------------
class ResidualBlockUp(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv1 = nn.Conv2d(in_ch,  out_ch, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.skip  = nn.Conv2d(in_ch,  out_ch, 1, 1, 0, bias=False)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn1(x))
        y = self.upsample(y)
        y = self.conv1(y)
        y = F.relu(self.bn2(y))
        y = self.conv2(y)
        return y + self.skip(self.upsample(x))


class ResidualBlockDown(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_sn: bool = True):
        super().__init__()
        Conv = lambda *a, **k: spectral_norm(nn.Conv2d(*a, **k)) if use_sn else nn.Conv2d(*a, **k)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.bn2 = nn.BatchNorm2d(in_ch)
        self.conv1 = Conv(in_ch, in_ch, 3, 1, 1, bias=False)
        self.conv2 = Conv(in_ch, out_ch, 3, 1, 1, bias=False)
        self.skip  = Conv(in_ch, out_ch, 1, 1, 0, bias=False)
        self.pool  = nn.AvgPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn1(x))
        y = self.conv1(y)
        y = F.relu(self.bn2(y))
        y = self.conv2(y)
        y = self.pool(y)
        return y + self.skip(self.pool(x))


# ----------------------------------------------------------------
#  Generator
# ----------------------------------------------------------------
class Generator(nn.Module):
    """
    Works for any square 2ᵏ resolution ≥ 32.
    For CIFAR-10 pass img_size=(3,32,32); for STL-10 default 96×96.
    """
    def __init__(self,
                 z_dim:    int                   = 128,
                 img_size: tuple[int,int,int]   = (3, 96, 96),
                 base_ch:  int                   = 64):
        super().__init__()
        c, h, w = img_size
        assert h == w and (h & (h-1)) == 0 and h >= 32, "Image must be 2^k square ≥ 32"

        # number of up/down blocks
        n_blocks = int(math.log2(h)) - 2              # 32→3, 64→4, 96→4, 128→5, ...
        dims     = [base_ch * (2**i) for i in reversed(range(n_blocks+1))]
        init_spatial = h // (2**n_blocks)

        # latent→feature map
        self.z_dim  = z_dim
        self.project = nn.Sequential(
            nn.Linear(z_dim, dims[0]*init_spatial*init_spatial),
            nn.ReLU(True),
            nn.BatchNorm1d(dims[0]*init_spatial*init_spatial),
        )

        # upsampling path (add SA at 32×32 & 64×64)
        blocks = []
        curr  = init_spatial
        for i in range(n_blocks):
            blocks.append(ResidualBlockUp(dims[i], dims[i+1]))
            curr *= 2
            if curr in (32, 64):
                blocks.append(SelfAttention(dims[i+1]))
        self.blocks = nn.Sequential(*blocks)

        # to RGB
        self.to_rgb = nn.Sequential(
            nn.Conv2d(dims[-1], c, 3, 1, 1, bias=False),
            nn.Tanh()
        )

        self.apply(weights_init)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b = z.size(0)
        # project & reshape
        x = self.project(z)
        feat_map_size = int(math.sqrt(x.shape[1] // (z.shape[1]//z.shape[1])))
        x = x.view(b, -1, feat_map_size, feat_map_size)
        # upsample
        x = self.blocks(x)
        # to RGB
        return self.to_rgb(x)


# ----------------------------------------------------------------
#  Discriminator
# ----------------------------------------------------------------
class Discriminator(nn.Module):
    """
    Spectral‐norm everywhere.  PatchGAN + global head.
    """
    def __init__(self,
                 img_size: tuple[int,int,int] = (3, 96, 96),
                 base_ch:  int                = 64,
                 is_critic: bool              = False):
        super().__init__()
        c, h, w = img_size
        assert h == w and (h & (h-1)) == 0 and h >= 32

        n_blocks = int(math.log2(h)) - 2
        dims     = [base_ch * (2**i) for i in range(n_blocks+1)]

        # down‐sampling path (SA at 64×64 & 32×32)
        blocks = []
        curr   = h
        in_ch  = c
        for out_ch in dims:
            blocks.append(ResidualBlockDown(in_ch, out_ch))
            curr //= 2
            in_ch = out_ch
            if curr in (64, 32):
                blocks.append(SelfAttention(out_ch))
        self.blocks = nn.Sequential(*blocks)

        # global head
        self.global_head = spectral_norm(nn.Conv2d(dims[-1], 1, curr, 1, 0))
        # 70×70 patch‐GAN head
        self.patch_head  = spectral_norm(nn.Conv2d(dims[-1], 1, 4, 1, 0))

        self.act_out = nn.Identity() if is_critic else nn.Sigmoid()
        self.flatten = nn.Flatten()

        self.apply(weights_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h   = self.blocks(x)
        g   = self.global_head(h).view(x.size(0), -1)
        p   = self.patch_head(h).mean([2,3])
        out = g + p
        return self.act_out(self.flatten(out)).view(-1)
