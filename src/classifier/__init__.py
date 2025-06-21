import torch
import torch.nn as nn
import timm


# ---------------------------------------------------------------------
# helper: transparent wrapper that upsamples small inputs to target_sz
# ---------------------------------------------------------------------
class _PadTo32AndForward(nn.Module):
    """If the incoming image is < 32 px, resize to 32×32 before the backbone."""
    def __init__(self, backbone: nn.Module, target_sz: int = 32):
        super().__init__()
        self.backbone   = backbone
        self.target_sz  = target_sz
        self.resize_bn  = nn.Upsample(
            size=target_sz, mode="bilinear", align_corners=False
        )

    def forward(self, x: torch.Tensor):
        if x.shape[-1] != self.target_sz:
            x = self.resize_bn(x)
        return self.backbone(x)


# ---------------------------------------------------------------------
# public factory
# ---------------------------------------------------------------------
def construct_classifier(params: dict, device: str | torch.device | None = None):
    """
    Build a TIMM backbone that also copes with 28×28 grey-scale inputs
    (MNIST / Fashion-MNIST) by up-sampling to 32×32 whenever required.

    params
    -------
    type       : TIMM model name — e.g. 'vit_tiny_patch16_32', 'convnext_tiny'.
                 Prefix with 'frozen_' to freeze the backbone.
    img_size   : tuple  (C, H, W)
    n_classes  : int
    """
    model_type  = params["type"]
    n_classes   = params["n_classes"]
    in_chans    = params.get("img_size", (3,))[0]          # infer C
    in_size     = params.get("img_size", (None, 28, 28))[1]  # infer H (28 for MNIST)

    # -- Optional "frozen_" prefix -------------------------------------
    freeze_backbone = False
    if model_type.startswith("frozen_"):
        freeze_backbone = True
        model_type = model_type[len("frozen_"):]
    # ------------------------------------------------------------------

    if model_type not in timm.list_models():
        raise ValueError(f"Unknown TIMM model '{model_type}'")

    # ---- decide whether the backbone takes an `img_size=` kw ----------
    accepts_img_size = any(
        token in model_type
        for token in ("vit", "deit", "beit", "tnt", "pit", "swin", "cait")
    )

    # ---- common kwargs -----------------------------------------------
    timm_kwargs = dict(
        pretrained=True,
        in_chans=in_chans,
        num_classes=n_classes,
    )

    wrap_resize = False
    if in_size < 32:
        # Up-sample inputs to 32×32 so ConvNeXt / ViT stages don’t break.
        wrap_resize = True
        if accepts_img_size:
            # ViT-family – tell timm to build weights for 32×32.
            timm_kwargs["img_size"] = 32

    # ---- create the backbone -----------------------------------------
    backbone = timm.create_model(model_type, **timm_kwargs)

    # ---- optionally freeze everything but the classifier head ---------
    if freeze_backbone:
        for p in backbone.parameters():
            p.requires_grad = False
        for p in backbone.get_classifier().parameters():
            p.requires_grad = True

    model: nn.Module = (
        _PadTo32AndForward(backbone, target_sz=32) if wrap_resize else backbone
    )

    # ---- move to device ----------------------------------------------
    if device is not None:
        model = model.to(device)

    return model
