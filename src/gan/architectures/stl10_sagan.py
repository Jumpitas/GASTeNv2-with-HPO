from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


# ───────────────────────── helpers ─────────────────────────
def pixel_norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Pixel-wise feature vector normalisation (StyleGAN)."""
    return x * torch.rsqrt(torch.mean(x**2, dim=1, keepdim=True) + eps)


def he(m: nn.Module) -> None:          # Kaiming-He init for Conv / Linear
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, a=0.2)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# ───────────────────── mapping network ─────────────────────
class Mapping(nn.Sequential):
    """8-layer MLP that maps Z→W (StyleGAN)."""
    def __init__(self, z_dim: int = 128, w_dim: int = 512, layers: int = 8):
        blocks = []
        for i in range(layers):
            inp = z_dim if i == 0 else w_dim
            blocks += [nn.Linear(inp, w_dim), nn.LeakyReLU(0.2)]
        super().__init__(*blocks)
        self.apply(he)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return pixel_norm(super().forward(z))


# ─────────────────── modulated convolution ─────────────────
class ModConv(nn.Module):
    """StyleGAN2 modulated (optionally demodulated) conv."""
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, demod: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_ch, in_ch, k, k))
        self.affine = nn.Linear(512, in_ch)
        self.k = k
        self.pad = k // 2
        self.demod = demod

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        b, c, h, w_ = x.shape
        style = self.affine(w).view(b, 1, c, 1, 1) + 1
        wgt = self.weight[None] * style
        if self.demod:
            d = torch.rsqrt((wgt ** 2).sum([2, 3, 4]) + 1e-8)
            wgt = wgt * d.view(b, -1, 1, 1, 1)
        wgt = wgt.view(-1, c, self.k, self.k)
        x = x.view(1, -1, h, w_)
        x = F.conv2d(x, wgt, padding=self.pad, groups=b)
        return x.view(b, -1, h, w_)


# ───────────────────── generator block ─────────────────────
class GBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.c1 = ModConv(in_ch, out_ch)
        self.c2 = ModConv(out_ch, out_ch)
        self.n1 = nn.Parameter(torch.zeros(1, out_ch, 1, 1))
        self.n2 = nn.Parameter(torch.zeros(1, out_ch, 1, 1))
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.act(self.c1(x, w) + self.n1 * torch.randn_like(x))
        x = self.act(self.c2(x, w) + self.n2 * torch.randn_like(x))
        return x


# ─────────────────── discriminator block ───────────────────
class DBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.c1 = spectral_norm(nn.Conv2d(in_ch, out_ch, 3, 1, 1))
        self.c2 = spectral_norm(nn.Conv2d(out_ch, out_ch, 3, 1, 1))
        self.skip = spectral_norm(nn.Conv2d(in_ch, out_ch, 1))
        self.pool = nn.AvgPool2d(2)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pool(self.act(self.c2(self.act(self.c1(x)))))
        return y + self.pool(self.skip(x))


# ───────────────────────── Generator ───────────────────────
class Generator(nn.Module):
    """
    StyleGAN-like generator for 1×128×128 chest-X-ray pairs.
    """
    def __init__(
        self,
        img_size: tuple[int, int, int] = (1, 128, 128),
        z_dim: int = 128,
        fmap: int = 64,
        **_
    ):
        super().__init__()
        self.z_dim = z_dim
        c, h, _ = img_size
        self.mapping = Mapping(z_dim)
        log_res = int(math.log2(h))           # 7 for 128
        self.const = nn.Parameter(torch.randn(1, fmap * 16, 4, 4))
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        in_ch = fmap * 16
        blocks, torgb = [], []
        for _ in range(log_res - 2):          # 5 blocks → 128×128
            out_ch = max(fmap, in_ch // 2)
            blocks.append(GBlock(in_ch, out_ch))
            torgb.append(ModConv(out_ch, c, 1, demod=False))
            in_ch = out_ch
        self.blocks = nn.ModuleList(blocks)
        self.torgb = nn.ModuleList(torgb)
        self.tanh = nn.Tanh()
        self.apply(he)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        w = self.mapping(z)
        x = self.const.expand(z.size(0), -1, -1, -1)
        rgb = None
        for blk, tor in zip(self.blocks, self.torgb):
            x = blk(x, w)
            rgb = tor(x, w) if rgb is None else self.upsample(rgb) + tor(x, w)
        return self.tanh(rgb)


# ──────────────────────── Discriminator ────────────────────
class Discriminator(nn.Module):
    """
    Spectral-norm residual discriminator with dynamic depth.
    """
    def __init__(
        self,
        img_size: tuple[int, int, int] = (1, 128, 128),
        fmap: int = 64,
        n_blocks: int = 5,
        is_critic: bool = False,
        **_
    ):
        super().__init__()
        c, h, _ = img_size
        max_b = max(int(math.log2(h)) - 2, 1)
        self.n_blocks = min(n_blocks, max_b)

        res = h // (2 ** self.n_blocks)
        layers = [spectral_norm(nn.Conv2d(c, fmap, 3, 1, 1))]
        in_ch = fmap
        for _ in range(self.n_blocks):
            out_ch = min(fmap * 16, in_ch * 2)
            layers.append(DBlock(in_ch, out_ch))
            in_ch = out_ch
            res //= 2

        self.features = nn.Sequential(*layers)
        self.final_fc = spectral_norm(nn.Linear(in_ch * res * res, 1))
        self.act_out = nn.Identity() if is_critic else nn.Sigmoid()
        self.apply(he)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        return self.act_out(self.final_fc(h.flatten(1))).view(-1)


# ───────────────────────── builders ────────────────────────
def build_cxr_g(z_dim: int = 128, base_ch: int = 64) -> Generator:
    return Generator((1, 128, 128), z_dim=z_dim, fmap=base_ch)


def build_cxr_d(base_ch: int = 64, critic: bool = False) -> Discriminator:
    return Discriminator((1, 128, 128), fmap=base_ch, is_critic=critic)