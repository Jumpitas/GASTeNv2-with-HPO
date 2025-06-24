from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ────────────────────────────────────────────────────────────────────
# helpers
# ────────────────────────────────────────────────────────────────────
class _ResizeForward(nn.Module):
    """Resize inputs to tgt_sz×tgt_sz (up or down) then run the backbone."""
    def __init__(self, backbone: nn.Module, tgt_sz: int):
        super().__init__()
        self.backbone = backbone
        self.tgt_sz   = tgt_sz
        self.resize   = nn.Upsample(size=tgt_sz, mode="bilinear",
                                    align_corners=False)

    def forward(self, x: torch.Tensor):
        if x.shape[-1] != self.tgt_sz:          # assumes square
            x = self.resize(x)
        return self.backbone(x)


class PooledMLP(nn.Module):
    """Adaptive-pooled MLP that works for any input resolution."""
    def __init__(self,
                 in_ch: int,
                 n_classes: int,
                 pool_sz: int = 32,
                 hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(pool_sz),              # C×S×S
            nn.Flatten(),                               # C·S²
            nn.Linear(in_ch * pool_sz * pool_sz, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, n_classes),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


# ────────────────────────────────────────────────────────────────────
# factory
# ────────────────────────────────────────────────────────────────────
def construct_classifier(params: dict,
                         device: str | torch.device | None = None) -> nn.Module:
    mtype       = params["type"].lower()
    n_cls       = params["n_classes"]
    C, H, W     = params.get("img_size", (3, 28, 28))
    hidden_dim  = params.get("nf", params.get("hidden_dim", 512))
    pool_sz     = params.get("mlp_pool", 32)            # MLP target size

    # ───────────── MLP branch ────────────────────────────────────────
    if mtype in {"mlp", "simple_mlp"}:
        backbone = PooledMLP(C, n_cls,
                             pool_sz=pool_sz,
                             hidden_dim=hidden_dim)
        model = backbone.to(device) if device else backbone
        return model

    # ───────────── TIMM branch (unchanged logic) ────────────────────
    freeze = mtype.startswith("frozen_")
    if freeze:
        mtype = mtype[len("frozen_"):]

    if mtype not in timm.list_models():
        raise ValueError(f"Unknown TIMM model '{mtype}'")

    patch_families = (
        "vit", "deit", "beit", "tnt", "pit",
        "swin", "cait", "pvt", "mvit", "eva",
        "mixer",
    )
    is_patch = any(tok in mtype for tok in patch_families)

    timm_kwargs = dict(pretrained=True, in_chans=C, num_classes=n_cls)

    if is_patch and H != 224:
        if H >= 64 and (H % 16 == 0):
            timm_kwargs.update(img_size=H, pretrained=False)
            tgt_sz = H
        else:
            tgt_sz = 224          # keep 224-px weights, resize inputs later
    else:
        tgt_sz = 224 if is_patch else H  # no resize for non-patch models

    backbone = timm.create_model(mtype, **timm_kwargs)

    if freeze:
        for p in backbone.parameters():
            p.requires_grad = False
        for p in backbone.get_classifier().parameters():
            p.requires_grad = True

    model = (_ResizeForward(backbone, tgt_sz)
             if (backbone.training and H != tgt_sz) else backbone)

    return model.to(device) if device else model