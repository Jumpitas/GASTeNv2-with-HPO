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
        nn.init.normal_(m.weight, 0, 0.02)
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


# ----------------------------------------------------------------
#  Self-Attention (unchanged – wrapped in checkpoint to save RAM)
# ----------------------------------------------------------------
class SelfAttention(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.f = spectral_norm(nn.Conv2d(ch,   ch // 8, 1))
        self.g = spectral_norm(nn.Conv2d(ch,   ch // 8, 1))
        self.h = spectral_norm(nn.Conv2d(ch,   ch // 2, 1))
        self.v = spectral_norm(nn.Conv2d(ch // 2, ch,      1))
        self.gamma = nn.Parameter(torch.zeros(1))

    def _core(self, x):
        b, c, h, w = x.size()
        f = self.f(x).view(b, -1, h * w)
        g = self.g(x).view(b, -1, h * w)
        beta = torch.softmax(torch.bmm(f.transpose(1, 2), g), dim=-1)          # B×HW×HW
        h_ = self.h(x).view(b, -1, h * w)
        o  = torch.bmm(h_, beta).view(b, c // 2, h, w)
        return x + self.gamma * self.v(o)

    def forward(self, x):
        return checkpoint(self._core, x)


# ----------------------------------------------------------------
#  Residual blocks
# ----------------------------------------------------------------
class ResidualBlockUp(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv1 = nn.Conv2d(in_ch,  out_ch, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.skip  = nn.Conv2d(in_ch,  out_ch, 1, 1, 0, bias=False)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x):
        y = F.relu(self.bn1(x))
        y = self.upsample(y)
        y = self.conv1(y)
        y = F.relu(self.bn2(y))
        y = self.conv2(y)
        return y + self.skip(self.upsample(x))


class ResidualBlockDown(nn.Module):
    def __init__(self, in_ch, out_ch, use_sn=True):
        super().__init__()
        Conv = lambda *a, **k: spectral_norm(nn.Conv2d(*a, **k)) if use_sn else nn.Conv2d(*a, **k)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.bn2 = nn.BatchNorm2d(in_ch)
        self.conv1 = Conv(in_ch, in_ch, 3, 1, 1, bias=False)
        self.conv2 = Conv(in_ch, out_ch, 3, 1, 1, bias=False)
        self.skip  = Conv(in_ch, out_ch, 1, 1, 0, bias=False)
        self.pool  = nn.AvgPool2d(2)

    def forward(self, x):
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
    Works for any square 2ᵏ resolution >= 32.
    For CIFAR-10 pass img_size=(3,32,32); for STL-10 default 96².
    """
    def __init__(self,
                 z_dim: int = 128,
                 img_size: tuple[int, int, int] = (3, 96, 96),
                 base_ch: int = 64):
        super().__init__()
        c, h, w = img_size
        assert h == w and h & (h - 1) == 0 and h >= 32, "Image must be 2^k square ≥ 32"

        n_blocks = int(math.log2(h)) - 2                 # 32→3, 64→4, 96→4, 128→5 …
        base_ch  = min(base_ch, 256) if h == 32 else base_ch  # lighter model for CIFAR-10

        # feature-map widths from coarse to fine
        dims = [base_ch * 2**i for i in reversed(range(n_blocks + 1))]
        self.init_spatial = h // 2 ** n_blocks

        self.project = nn.Sequential(
            nn.Linear(z_dim, dims[0] * self.init_spatial ** 2),
            nn.ReLU(True),
            nn.BatchNorm1d(dims[0] * self.init_spatial ** 2),
        )

        blocks, curr = [], self.init_spatial
        for i in range(n_blocks):
            blocks.append(ResidualBlockUp(dims[i], dims[i + 1]))
            curr *= 2
            if curr in (32, 64):                         # add SA only if map ≥ 32 px
                blocks.append(SelfAttention(dims[i + 1]))
        self.blocks = nn.Sequential(*blocks)

        self.to_rgb = nn.Sequential(
            nn.Conv2d(dims[-1], c, 3, 1, 1, bias=False),
            nn.Tanh()
        )
        self.z_dim = z_dim
        self.apply(weights_init)

    def forward(self, z):
        b = z.size(0)
        x = self.project(z).view(b, -1, self.init_spatial, self.init_spatial)
        x = self.blocks(x)
        return self.to_rgb(x)


# ----------------------------------------------------------------
#  Discriminator
# ----------------------------------------------------------------
class Discriminator(nn.Module):
    """Spectral-Norm everywhere (as in BigGAN-lite)."""
    def __init__(self,
                 img_size: tuple[int, int, int] = (3, 96, 96),
                 base_ch: int = 64,
                 is_critic: bool = False):
        super().__init__()
        c, h, w = img_size
        assert h == w and h & (h - 1) == 0 and h >= 32

        n_blocks = int(math.log2(h)) - 2
        base_ch  = min(base_ch, 256) if h == 32 else base_ch

        dims = [base_ch * 2**i for i in range(n_blocks + 1)]
        blocks, curr, in_ch = [], h, c
        for out_ch in dims:
            blocks.append(ResidualBlockDown(in_ch, out_ch, use_sn=True))
            curr //= 2
            in_ch = out_ch
            if curr in (64, 32):                         # SA mirrors the generator
                blocks.append(SelfAttention(out_ch))
        self.blocks = nn.Sequential(*blocks)

        self.final = spectral_norm(
            nn.Conv2d(dims[-1], 1, curr, 1, 0, bias=False)
        )
        self.flatten = nn.Flatten()
        self.act_out = nn.Identity() if is_critic else nn.Sigmoid()
        self.apply(weights_init)

    def forward(self, x):
        h = self.blocks(x)
        h = self.final(h)
        return self.act_out(self.flatten(h)).view(-1)