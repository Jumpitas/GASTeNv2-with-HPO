import torch
import torch.nn as nn
import timm


def construct_classifier(params, device=None):
    """
    Construct a classifier solely from TIMM models.

    params:
      - type:      timm model name (e.g. 'resnet18', 'convnext_tiny', ...)
                   or prefixed with 'frozen_' to freeze all backbone weights
      - img_size:  tuple (C, H, W)
      - n_classes: number of output classes
    """
    model_type = params['type']
    n_classes = params['n_classes']
    # infer input channels from img_size, default to 3
    in_chans = params.get('img_size', (3,))[0]

    # handle optional freezing
    freeze_backbone = False
    if model_type.startswith('frozen_'):
        freeze_backbone = True
        model_type = model_type[len('frozen_'):]

    # validate model name
    if model_type not in timm.list_models():
        raise ValueError(f"Unknown TIMM model '{model_type}'")

    # create pretrained model with specified input channels and output classes
    model = timm.create_model(
        model_type,
        pretrained=True,
        in_chans=in_chans,
        num_classes=n_classes
    )

    # optionally freeze backbone parameters
    if freeze_backbone:
        # freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
        # unfreeze classifier head
        # get_classifier() returns the final classification layer or sequence
        classifier_head = model.get_classifier()
        if isinstance(classifier_head, nn.Sequential):
            for param in classifier_head.parameters():
                param.requires_grad = True
        else:
            for param in classifier_head.parameters():
                param.requires_grad = True

    # move model to device if provided
    if device is not None:
        model = model.to(device)

    return model
