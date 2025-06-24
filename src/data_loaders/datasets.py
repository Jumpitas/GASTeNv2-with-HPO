import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
from datasets import load_dataset as hf_load_dataset
from PIL import Image

# 1. Standard torchvision loaders
# ────────────────────────────────────────────────────────────────────

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


# 2. Chest-XRay loader (unchanged)
# ────────────────────────────────────────────────────────────────────

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


# 3. Full ImageNet via Hugging-Face (non-streaming)
# ────────────────────────────────────────────────────────────────────

def get_imagenet(dataroot, train=True):
    """
    Download & cache the full ILSVRC2012 dataset via HF.  The first run will
    fetch ~150GB; subsequent runs will load from disk.
    """
    split = "train" if train else "validation"
    hf_ds = hf_load_dataset(
        "imagenet-1k",
        split=split,
        cache_dir=dataroot,
        use_auth_token=False  # explicitly disable any token param
    )

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    class HFImageNet(Dataset):
        def __init__(self, hf_dataset, transform):
            self.ds        = hf_dataset
            self.transform = transform

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, idx):
            ex  = self.ds[idx]
            img = ex["image"].convert("RGB")
            img = self.transform(img)
            lbl = ex["label"]
            return img, lbl

        @property
        def targets(self):
            return torch.tensor([self.ds[i]["label"] for i in range(len(self.ds))])

    return HFImageNet(hf_ds, transform)