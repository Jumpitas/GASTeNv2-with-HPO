import torch
import torch.nn as nn
import timm


# ------------------------------------------------------------------
# helper: up-sample very small inputs to 32×32 before the backbone
# ------------------------------------------------------------------
class _PadAndForward(nn.Module):
    """Upscale inputs (< tgt_sz) to tgt_sz×tgt_sz and run the backbone."""
    def __init__(self, backbone: nn.Module, tgt_sz: int = 32):
        super().__init__()
        self.backbone  = backbone
        self.tgt_sz    = tgt_sz
        self.resizer   = nn.Upsample(size=tgt_sz,
                                     mode="bilinear",
                                     align_corners=False)

    def forward(self, x: torch.Tensor):
        if x.shape[-1] < self.tgt_sz:          # H (== W)  – for square inputs
            x = self.resizer(x)
        return self.backbone(x)


# ------------------------------------------------------------------
# public factory
# ------------------------------------------------------------------
def construct_classifier(params: dict,
                         device: str | torch.device | None = None) -> nn.Module:
    """
    Build a TIMM classifier that copes with many datasets/sizes:

    * 28×28   (MNIST / Fashion-MNIST)      – up-sample to 32
    * 32×32   (CIFAR)                      – native
    * 96×96   (STL-10)                     – native
    * 128×128 (Chest X-ray)                – native / ViT rebuilt
    * 224×224 (ImageNet)                   – native
    """
    mtype      = params["type"]                  # possibly 'frozen_*'
    n_cls      = params["n_classes"]
    C, H, _    = params.get("img_size", (3, 28, 28))

    # ------------ optional "frozen_" prefix --------------------------
    freeze = False
    if mtype.startswith("frozen_"):
        freeze = True
        mtype  = mtype[len("frozen_"):]
    # -----------------------------------------------------------------

    if mtype not in timm.list_models():
        raise ValueError(f"Unknown TIMM model '{mtype}'")

    # ---- does this backbone support an explicit img_size kwarg? -----
    vit_families = ("vit", "deit", "beit", "tnt", "pit",
                    "swin", "cait", "pvt", "mvit", "eva")
    accepts_img_size = any(tok in mtype for tok in vit_families)

    timm_kwargs = dict(
        pretrained=True,
        in_chans=C,
        num_classes=n_cls,
    )

    # Re-build positional embeddings if needed (e.g. 128-pixel chest X-ray)
    if accepts_img_size and H != 224:
        if H % 8 != 0:                         # patch-based models need /8
            raise ValueError(
                f"{mtype} requires img_size multiple of the patch size; "
                f"you passed {H}."
            )
        timm_kwargs["img_size"] = H

    # -----------------------------------------------------------------
    backbone: nn.Module = timm.create_model(mtype, **timm_kwargs)
    # -----------------------------------------------------------------

    # Optionally freeze the backbone except the classifier head
    if freeze:
        for p in backbone.parameters():
            p.requires_grad = False
        for p in backbone.get_classifier().parameters():
            p.requires_grad = True

    # Up-sample only if really tiny (<32 px)
    model: nn.Module = (
        _PadAndForward(backbone, tgt_sz=32) if H < 32 else backbone
    )

    if device is not None:
        model = model.to(device)

    return model
