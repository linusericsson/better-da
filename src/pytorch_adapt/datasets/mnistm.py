import os

import torch
from torchvision.datasets import MNIST

from .base_dataset import BaseDownloadableDataset
from .utils import check_length, check_split


class MNISTM(BaseDownloadableDataset):
    """
    The dataset used in "Domain-Adversarial Training of Neural Networks".
    It consists of colored MNIST digits.

    Extends [BaseDownloadableDataset][pytorch_adapt.datasets.BaseDownloadableDataset],
    so the dataset can be downloaded by setting ```download=True``` when
    initializing.
    """

    url = "https://cornell.box.com/shared/static/jado7quprg6hzzdubvwzh9umr75damwi"
    filename = "mnist_m.tar.gz"
    md5 = "859df31c91afe82e80e5012ba928f279"

    def __init__(self, root: str, split: str, transform=None, **kwargs):
        """
        Arguments:
            root: The dataset must be located at ```<root>/mnist_m```
            split: Which data split to use (train/val/trainval/test).
            transform: The image transform applied to each sample.
        """
        self.split = check_split(split)
        super().__init__(root=root, **kwargs)
        self.transform = transform

    def set_paths_and_labels(self, root):
        labels_file = os.path.join(root, "mnist_m", f"mnist_m_{self.split}_labels.txt")
        split_dir = "train" if self.split in ["train", "val", "trainval"] else "test"
        img_dir = os.path.join(root, "mnist_m", f"mnist_m_{split_dir}")
        with open(labels_file) as f:
            content = [line.rstrip().split(" ") for line in f]
        self.img_paths = [os.path.join(img_dir, x[0]) for x in content]
        check_length(self, {"train": 49001, "val": 10000, "trainval": 59001, "test": 9001}[self.split])
        self.labels = [int(x[1]) for x in content]


class BaseMNIST(MNIST):

    def __init__(self, root: str, split: str, transform=None, **kwargs):
        """
        Arguments:
            root: The dataset must be located at ```<root>/mnist_m```
            split: Which data split to use (train/val/trainval/test).
            transform: The image transform applied to each sample.
        """
        self.split = check_split(split)
        super().__init__(root=os.path.join(root, "MNIST"), train=(split != "test"), transform=transform, download=True)
        self.transform = transform

        if split in ["train", "val"]:
            rng = torch.Generator()
            rng.manual_seed(0)
            perm = torch.randperm(len(self.data))
            split_indices = {"train": perm[:-10000], "val": perm[-10000:]}
            self.data, self.targets = [self.data[i] for i in split_indices[self.split]], [self.targets[i] for i in split_indices[self.split]]
