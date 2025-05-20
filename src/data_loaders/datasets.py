import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from datasets import load_dataset as hf_load_dataset,  DownloadConfig
from torch.utils.data import IterableDataset

def get_mnist(dataroot, train=True):
    dataset = torchvision.datasets.MNIST(
        root=dataroot, download=True, train=train,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]))
    return dataset


def get_fashion_mnist(dataroot, train=True):
    dataset = torchvision.datasets.FashionMNIST(
        root=dataroot, download=True, train=train,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]))
    return dataset


def get_cifar10(dataroot, train=True):
    dataset = torchvision.datasets.CIFAR10(
        root=dataroot, download=True, train=train,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5)),
        ]))
    return dataset


def get_stl10(dataroot, train=True):
    split = "train" if train else "test"
    dataset = torchvision.datasets.STL10(
        root=dataroot, download=True, split=split,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5)),
        ]))
    return dataset


def get_chest_xray(dataroot, train=True):
    split = "train" if train else "test"
    hf_dataset = hf_load_dataset(
        "keremberke/chest-xray-classification",
        "full",
        split=split,
        cache_dir=dataroot
    )

    transform = transforms.Compose([
        # make sure the grayscale PIL→RGB
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5)),
    ])

    class ChestXrayDataset(Dataset):
        def __init__(self, hf_dataset, transform=None):
            self.dataset = hf_dataset
            self.transform = transform

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            item = self.dataset[idx]
            img = item["image"]
            lbl = item["labels"]
            if self.transform:
                img = self.transform(img)
            return img, lbl

        @property
        def targets(self):
            return torch.tensor([self.dataset[i]["labels"]
                                 for i in range(len(self.dataset))])

    return ChestXrayDataset(hf_dataset, transform=transform)

def get_imagenet(dataroot, train=True):
    split = "train" if train else "validation"
    # stream-only so you don’t download all 150 GB
    hf_ds = hf_load_dataset(
        "imagenet-1k",
        split=split,
        streaming=True,
        cache_dir=dataroot,
        trust_remote_code=True,
    )

    # only keep the two labels you care about
    TARGETS = {281, 282}
    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),
    ])

    # pull out just those examples into a list
    examples = []
    for example in hf_ds:
        lbl = example["label"]
        if lbl in TARGETS:
            img = transform(example["image"])
            # remap labels to 0/1 if you like, or leave as 281/282 and
            # let your BinaryDataset wrapper convert them
            examples.append((img, lbl))

    class ImageNetBinaryDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    return ImageNetBinaryDataset(examples)