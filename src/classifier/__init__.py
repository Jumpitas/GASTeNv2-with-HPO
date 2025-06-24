from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ────────────────────────────────────────────────────────────────────
# helpers
# ────────────────────────────────────────────────────────────────────
class _PadAndForward(nn.Module):
    """Upscale inputs (< tgt_sz) to tgt_sz×tgt_sz then run the backbone."""
    def __init__(self, backbone: nn.Module, tgt_sz: int):
        super().__init__()
        self.backbone = backbone
        self.tgt_sz   = tgt_sz
        self.resize   = nn.Upsample(size=tgt_sz,
                                    mode="bilinear",
                                    align_corners=False)

    def forward(self, x: torch.Tensor):
        if x.shape[-1] < self.tgt_sz:              # assumes square images
            x = self.resize(x)
        return self.backbone(x)


class SimpleMLP(nn.Module):
    """Two-hidden-layer MLP classifier."""
    def __init__(self,
                 in_shape: tuple[int, int, int],   # (C, H_eff, W_eff)
                 n_classes: int,
                 hidden_dim: int = 512):
        super().__init__()
        C, H, W = in_shape
        in_dim  = C * H * W
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, hidden_dim),
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
def construct_classifier(
    params: dict,
    device: str | torch.device | None = None,
) -> nn.Module:
    """
    Build either
      • a TIMM backbone (CNN / ViT / Swin / Mixer …)
      • or a lightweight SimpleMLP when params["type"] in {"mlp","simple_mlp"}
    """

    mtype      = params["type"].lower()
    n_cls      = params["n_classes"]
    C, H, W    = params.get("img_size", (3, 28, 28))
    hidden_dim = params.get("hidden_dim", 512)   # only read for MLP

    # ───────────── simple MLP branch ────────────────────────────────
    if mtype in {"mlp", "simple_mlp"}:
        tgt_sz = params.get("min_size", 32)      # upsample floor
        H_eff  = max(H, tgt_sz)                  # effective size after resize
        W_eff  = max(W, tgt_sz)

        backbone = SimpleMLP((C, H_eff, W_eff), n_cls, hidden_dim=hidden_dim)
        model    = _PadAndForward(backbone, tgt_sz) if H < tgt_sz else backbone
        return model.to(device) if device else model

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
    is_patch_model = any(tok in mtype for tok in patch_families)

    timm_kwargs = dict(
        pretrained=True,
        in_chans=C,
        num_classes=n_cls,
    )

    # rebuild patch models for custom img_size if multiples of 16
    if is_patch_model and H != 224:
        if H >= 64 and (H % 16 == 0):
            timm_kwargs.update(img_size=H, pretrained=False)
        else:
            H = 224                              # keep 224-px weights

    backbone: nn.Module = timm.create_model(mtype, **timm_kwargs)

    if freeze:
        for p in backbone.parameters():
            p.requires_grad = False
        for p in backbone.get_classifier().parameters():
            p.requires_grad = True

    tgt_sz = 224 if is_patch_model else 32
    model  = (_PadAndForward(backbone, tgt_sz=tgt_sz)
              if params["img_size"][1] < tgt_sz else backbone)

    return model.to(device) if device else model