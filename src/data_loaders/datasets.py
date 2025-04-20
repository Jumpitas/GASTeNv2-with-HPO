import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from datasets import load_dataset as hf_load_dataset  # Hugging Face data_loaders library


def get_mnist(dataroot, train=True):
    dataset = torchvision.datasets.MNIST(
        root=dataroot, download=True, train=train,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,)),
        ]))
    return dataset


def get_fashion_mnist(dataroot, train=True):
    dataset = torchvision.datasets.FashionMNIST(
        root=dataroot, download=True, train=train,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,)),
        ]))
    return dataset


def get_cifar10(dataroot, train=True):
    dataset = torchvision.datasets.CIFAR10(
        root=dataroot, download=True, train=train,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                             (0.5, 0.5, 0.5)),
        ]))
    return dataset


def get_stl10(dataroot, train=True):
    split = "train" if train else "test"
    dataset = torchvision.datasets.STL10(
        root=dataroot, download=True, split=split,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                             (0.5, 0.5, 0.5)),
        ]))
    return dataset


def get_chest_xray(dataroot, train=True):
    split = "train" if train else "test"
    hf_dataset = hf_load_dataset("keremberke/chest-xray-classification", "full", split=split, cache_dir=dataroot)

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])

    class ChestXrayDataset(Dataset):
        def __init__(self, hf_dataset, transform=None):
            self.dataset = hf_dataset
            self.transform = transform

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            item = self.dataset[idx]
            image = item["image"]
            label = item["labels"]
            if self.transform is not None:
                image = self.transform(image)
            return image, label

        @property
        def targets(self):
            # Compute targets from the dataset; assumes labels are numeric.
            return torch.tensor([self.dataset[i]["labels"] for i in range(len(self.dataset))])

    return ChestXrayDataset(hf_dataset, transform=transform)
