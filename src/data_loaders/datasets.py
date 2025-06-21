import os
import hashlib
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from datasets import load_dataset as hf_load_dataset

# ------------------------------------------------------------------
# 1.  Standard torchvision loaders (unchanged)
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
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5)),
        ])
    )

def get_stl10(dataroot, train=True):
    split = "train" if train else "test"
    return torchvision.datasets.STL10(
        root=dataroot, download=True, split=split,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5)),
        ])
    )

# ------------------------------------------------------------------
# 2.  Chest-X-ray loader (unchanged)
# ------------------------------------------------------------------
def get_chest_xray(dataroot, train=True):
    split = "train" if train else "test"
    hf_dataset = hf_load_dataset(
        "keremberke/chest-xray-classification", "full",
        split=split, cache_dir=dataroot
    )

    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5)),
    ])

    class ChestXrayDataset(Dataset):
        def __init__(self, hf_dataset, transform):
            self.dataset   = hf_dataset
            self.transform = transform

        def __len__(self): return len(self.dataset)

        def __getitem__(self, idx):
            item = self.dataset[idx]
            img  = self.transform(item["image"])
            lbl  = item["labels"]
            return img, lbl

        @property
        def targets(self):
            return torch.tensor([self.dataset[i]["labels"]
                                 for i in range(len(self.dataset))])

    return ChestXrayDataset(hf_dataset, transform)

# ------------------------------------------------------------------
# 3.  ImageNet two-class loader (new, disk-cached)
# ------------------------------------------------------------------
def _ensure_local_imagenet_pair(root_dir: str,
                                split: str,
                                targets: dict[int, str],
                                max_per_class: int = 2048):
    """
    Download the two requested ImageNet classes once and store them under
    <root_dir>/imagenet_pair/<split>/<synset>/img_XXXX.jpg
    """
    pair_root = Path(root_dir) / "imagenet_pair" / split
    already = all((pair_root / syn).is_dir() for syn in targets.values())
    if already:
        return pair_root

    print(f"[ImageNet-pair] Preparing {split} split locally …")
    hf_ds = hf_load_dataset(
        "imagenet-1k",
        split="train" if split == "train" else "validation",
        streaming=True,
        trust_remote_code=True,
        cache_dir=root_dir
    )

    transform_jpeg = lambda img: img.convert("RGB")
    counters = {k: 0 for k in targets}

    pair_root.mkdir(parents=True, exist_ok=True)
    for ex in hf_ds:
        lbl = ex["label"]
        if lbl not in targets:
            continue
        if counters[lbl] >= max_per_class:
            if all(counters[k] >= max_per_class for k in counters):
                break
            continue

        syn = targets[lbl]
        dst_dir = pair_root / syn
        dst_dir.mkdir(parents=True, exist_ok=True)

        img = transform_jpeg(ex["image"])
        # deterministic hash for idempotency
        h = hashlib.md5(img.tobytes()).hexdigest()[:10]
        img.save(dst_dir / f"{syn}_{h}.jpg", "JPEG", quality=90)
        counters[lbl] += 1

    print("[ImageNet-pair] Saved locally to", pair_root)
    return pair_root

def get_imagenet(dataroot, train=True):
    """
    Return an ImageFolder containing exactly *two* ImageNet classes.
    The first run downloads the images; subsequent runs just load them.
    """
    # ------ customise the pair here ------------------------------------
    TARGETS = {                     # {label-id : WordNet synset}
        207: "n02099601",           # golden-retriever
        208: "n02099712",           # labrador-retriever
    }
    MAX_PER_CLASS = 2048            # download up to 2 048 images per class
    # -------------------------------------------------------------------

    split = "train" if train else "val"
    local_root = _ensure_local_imagenet_pair(
        dataroot, split, TARGETS, MAX_PER_CLASS
    )

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),
    ])
    dataset = torchvision.datasets.ImageFolder(local_root, transform=transform)
    return dataset
