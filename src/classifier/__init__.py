import torch.nn as nn
import timm
from .simple_cnn import Classifier as SimpleCNN
from .my_mlp import Classifier as MyMLP

def construct_classifier(params, device=None):
    t = params['type']
    n_classes = params['n_classes']

    if t == 'cnn':
        C = SimpleCNN(params['img_size'], [params['nf'], params['nf'] * 2], n_classes)

    elif t == 'mlp':
        C = MyMLP(params['img_size'], n_classes, params['nf'])

    elif t == 'resnet18_frozen':
        # freeze all layers except final fc
        model = timm.create_model('resnet18', pretrained=True, num_classes=n_classes)
        for p in model.parameters():
            p.requires_grad = False
        in_feats = model.get_classifier().in_features
        model.fc = nn.Linear(in_feats, n_classes)
        C = model

    elif t == 'resnet18_finetune':
        # full fine‐tuning
        C = timm.create_model('resnet18', pretrained=True, num_classes=n_classes)

    elif t == 'vit_s':
        # ViT-Small (patch 32) at 224×224
        C = timm.create_model('vit_small_patch32_224', pretrained=True, num_classes=n_classes)

    elif t == 'convnext_t':
        # ConvNeXt-Tiny
        C = timm.create_model('convnext_tiny', pretrained=True, num_classes=n_classes)

    else:
        raise ValueError(f"Unknown classifier type: {t}")

    return C.to(device)
