from __future__ import annotations

import torch
import torch.nn as nn
import timm


# ----------------------------------------------------------------------
# helper – resize tiny inputs before the backbone
# ----------------------------------------------------------------------
class _PadAndForward(nn.Module):
    """Upscale inputs (< tgt_sz) to tgt_sz×tgt_sz and run the backbone."""
    def __init__(self, backbone: nn.Module, tgt_sz: int):
        super().__init__()
        self.backbone = backbone
        self.tgt_sz   = tgt_sz
        self.resize   = nn.Upsample(size=tgt_sz,
                                    mode="bilinear",
                                    align_corners=False)

    def forward(self, x: torch.Tensor):
        if x.shape[-1] < self.tgt_sz:          # assumes square images
            x = self.resize(x)
        return self.backbone(x)


# ----------------------------------------------------------------------
# universal TIMM factory
# ----------------------------------------------------------------------
def construct_classifier(
    params: dict,
    device: str | torch.device | None = None,
) -> nn.Module:
    """
    Works with CNNs, ViTs/Swin/etc., and MLP-Mixers on 28–224-px images.
    """

    mtype: str = params["type"]                 # may start with 'frozen_'
    n_cls: int = params["n_classes"]
    C, H, _ = params.get("img_size", (3, 28, 28))

    # -------------------------------------------------- optional “frozen_” prefix
    freeze = mtype.startswith("frozen_")
    if freeze:
        mtype = mtype[len("frozen_") :]

    if mtype not in timm.list_models():
        raise ValueError(f"Unknown TIMM model '{mtype}'")

    # families that expose an img_size kwarg
    patch_families = (
        "vit", "deit", "beit", "tnt", "pit",
        "swin", "cait", "pvt", "mvit", "eva",
        "mixer",
    )
    is_patch_model = any(tok in mtype for tok in patch_families)

    # -------------------------------------------------- timm kwargs
    timm_kwargs: dict = dict(
        pretrained=True,
        in_chans=C,
        num_classes=n_cls,
    )

    # decide whether to *rebuild* the patch model or keep 224-px weights
    if is_patch_model and H != 224:
        can_rebuild = (H >= 64) and (H % 16 == 0)
        if can_rebuild:
            timm_kwargs.update(img_size=H, pretrained=False)
        else:
            # keep pretrained 224-px geometry; inputs will be upsampled later
            H = 224   # so downstream logic knows the effective backbone size

    # -------------------------------------------------- build the backbone
    backbone: nn.Module = timm.create_model(mtype, **timm_kwargs)

    # -------------------------------------------------- optionally freeze
    if freeze:
        for p in backbone.parameters():
            p.requires_grad = False
        for p in backbone.get_classifier().parameters():
            p.requires_grad = True

    # -------------------------------------------------- handle tiny inputs
    tgt_sz = 224 if is_patch_model else 32
    model: nn.Module = (
        _PadAndForward(backbone, tgt_sz=tgt_sz) if params["img_size"][1] < tgt_sz else backbone
    )

    # -------------------------------------------------- move to device
    if device is not None:
        model = model.to(device)

    return model
