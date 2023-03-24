import os

import torch
from torchvision import datasets as torch_datasets

from .base_dataset import BaseDataset, BaseDownloadableDataset
from .utils import check_img_paths, check_length, check_split


class VisDA2017ClassificationFull(BaseDataset):
    """
    A large dataset consisting of 12 classes in 3 domains:
    train (synthetic), validation (real), test (real).
    """

    def __init__(self, root: str, domain: str, transform):
        """
        Arguments:
            root: The dataset must be located at ```<root>/VisDA2017```
            domain: One of ```"train", "validation", "test"```.
            transform: The image transform applied to each sample.
        """

        super().__init__(domain=domain)
        self.transform = transform
        self.dataset = torch_datasets.ImageFolder(
            os.path.join(root, "classification", domain,), transform=self.transform
        )
        check_length(self, {"train": 152397, "validation": 55388, "test": 72372}[domain])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class VisDA2017Classification(BaseDownloadableDataset):
    """
    A custom train/test split of [VisDA2017ClassificationFull][pytorch_adapt.datasets.VisDA2017ClassificationFull].

    Extends [BaseDownloadableDataset][pytorch_adapt.datasets.BaseDownloadableDataset],
    so the dataset can be downloaded by setting ```download=True``` when
    initializing.
    """

    def __init__(self, root: str, domain: str, split: str, transform=None, **kwargs):
        """
        Arguments:
            root: The dataset must be located at ```<root>/VisDA2017```
            domain: One of ```"train", "validation", "test"```.
            train: Whether or not to use the training set.
            transform: The image transform applied to each sample.
        """
        self.split = check_split(split)
        super().__init__(root=root, domain=domain, **kwargs)
        self.transform = transform

    def set_paths_and_labels(self, root):
        split = "train" if self.split in ["train", "val", "trainval"] else self.split
        labels_file = os.path.join(root, "VisDA2017", 'classification', self.domain, f"{split}.txt")
        img_dir = os.path.join(root, "VisDA2017", 'classification')

        with open(labels_file) as f:
            content = [line.rstrip().split(" ") for line in f]

        n = len(content)
        torch.manual_seed(0)
        perm = torch.randperm(n)
        if split in ["train", "test"]:
            idx = perm[:5000]
        elif split == "val":
            idx = perm(n)[5000:10000]
        elif split == "trainval":
            idx = perm(n)[:10000]
        content = [content[i] for i in idx]

        self.img_paths = [os.path.join(img_dir, x[0]) for x in content]

        check_img_paths(img_dir, self.img_paths, self.domain)
        check_length(
            self,
            {
                #"train": {"train": 121917, "test": 30480}[name],
                #"validation": {"train": 44310, "test": 11078}[name],
                #"test": {"train": 57897, "test": 14475}[name],
                "train": {"train": 5000, "val": 5000, "trainval": 10000, "test": 5000}[split],
                "validation": {"train": 5000, "val": 5000, "trainval": 10000, "test": 5000}[split],
                "test": {"train": 5000, "val": 5000, "trainval": 10000, "test": 5000}[split],
            }[self.domain],
        )
        self.labels = [int(x[1]) for x in content]
