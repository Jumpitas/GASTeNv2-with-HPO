import os
import hashlib
from pathlib import Path

import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
from datasets import load_dataset as hf_load_dataset
from PIL import Image

# ------------------------------------------------------------------
# 1.  Standard torchvision loaders
# ------------------------------------------------------------------
def get_mnist(dataroot, train=True):
    return torchvision.datasets.MNIST(
        root=dataroot, download=True, train=train,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
    )

def get_fashion_mnist(dataroot, train=True):
    return torchvision.datasets.FashionMNIST(
        root=dataroot, download=True, train=train,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
    )

def get_cifar10(dataroot, train=True):
    return torchvision.datasets.CIFAR10(
        root=dataroot, download=True, train=train,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
        ])
    )

def get_stl10(dataroot, train=True):
    split = "train" if train else "test"
    return torchvision.datasets.STL10(
        root=dataroot, download=True, split=split,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
        ])
    )

# ------------------------------------------------------------------
# 2.  Chest-Xray loader
# ------------------------------------------------------------------
def get_chest_xray(dataroot, train=True):
    split = "train" if train else "test"
    hf_ds = hf_load_dataset(
        "keremberke/chest-xray-classification", "full",
        split=split, cache_dir=dataroot
    )
    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
    ])
    class ChestXrayDataset(Dataset):
        def __init__(self, hf_dataset, transform):
            self.dataset   = hf_dataset
            self.transform = transform
        def __len__(self):
            return len(self.dataset)
        def __getitem__(self, idx):
            item = self.dataset[idx]
            img  = self.transform(item["image"])
            lbl  = item["labels"]
            return img, lbl
        @property
        def targets(self):
            return torch.tensor([self.dataset[i]["labels"]
                                 for i in range(len(self.dataset))])
    return ChestXrayDataset(hf_ds, transform)

# ------------------------------------------------------------------
# 3.  Two‐class ImageNet loader w/ local cache
# ------------------------------------------------------------------
def _ensure_local_imagenet_pair(root_dir: str,
                                split: str,
                                targets: dict[int,str],
                                max_per_class: int = 2048):
    pair_root = Path(root_dir)/"imagenet_pair"/split
    # if both class‐dirs already exist, assume done
    if all((pair_root/wn).is_dir() for wn in targets.values()):
        return pair_root

    print(f"[ImageNet] caching split={split} to {pair_root}")
    ds = hf_load_dataset(
        "imagenet-1k",
        split="train" if split=="train" else "validation",
        streaming=True,
        cache_dir=root_dir
    )
    counts = {lbl:0 for lbl in targets}
    pair_root.mkdir(parents=True, exist_ok=True)

    for ex in ds:
        lbl = ex["label"]
        if lbl not in counts: continue
        if counts[lbl] >= max_per_class:
            if all(c>=max_per_class for c in counts.values()):
                break
            else:
                continue

        img = ex["image"].convert("RGB")
        syn = targets[lbl]
        out_dir = pair_root/syn
        out_dir.mkdir(exist_ok=True, parents=True)

        # stable filename
        h = hashlib.md5(img.tobytes()).hexdigest()[:10]
        img.save(out_dir/f"{syn}_{h}.jpg","JPEG",quality=90)
        counts[lbl] += 1

    print("[ImageNet] cached successfully.")
    return pair_root

def get_imagenet(dataroot, train=True):
    """
    Loads exactly two ImageNet classes via ImageFolder,
    caching them under <dataroot>/imagenet_pair/<split>/<synset>/
    on first run.
    """
    TARGETS = {207:"n02099601", 208:"n02099712"}  # golden vs labrador retriever
    split   = "train" if train else "validation"
    local   = _ensure_local_imagenet_pair(dataroot, split, TARGETS)

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),
                             (0.229,0.224,0.225)),
    ])
    return torchvision.datasets.ImageFolder(local, transform=transform)
