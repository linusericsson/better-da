import os
import pickle as pkl

import numpy as np
from PIL import Image

import torch
from torchvision import transforms

from .base_dataset import BaseDownloadableDataset
from .utils import check_length, check_split


class MNISTMR():
    """
    Regression version of MNIST-M.
    It consists of colored MNIST digits.

    Extends [BaseDownloadableDataset][pytorch_adapt.datasets.BaseDownloadableDataset],
    so the dataset can be downloaded by setting ```download=True``` when
    initializing.
    """

    def __init__(self, root: str, domain: str, split: str, transform=None, **kwargs):
        """
        Arguments:
            root: The dataset must be located at ```<root>/mnist_m_r```
            domain: one of mnist/mnistm
            split: Which data split to use (train/val/trainval/test).
            transform: The image transform applied to each sample.
        """
        self.domain = domain
        self.split = check_split(split)
        self.transform = transform
        self.to_pil = transforms.ToPILImage()

        data = pkl.load(open(os.path.join(root, "mnist_m_r", f"{self.domain}.pkl"), "rb"))
        lengths = {"train": 5000, "val": 5000, "trainval": 10000, "test": 5000,
                   "trainlabels": 5000, "vallabels": 5000, "trainvallabels": 10000, "testlabels": 5000}
        #lengths = {"train": 50000, "val": 10000, "trainval": 60000, "test": 10000,
        #           "trainlabels": 50000, "vallabels": 10000, "trainvallabels": 60000, "testlabels": 10000}
        data = {key: values[:lengths[key]] for key, values in data.items()}
        self.imgs = data[self.split]
        self.labels = data[f"{self.split}labels"]
        check_length(
            self,
            {
                "mnist": {"train": 5000, "val": 5000, "trainval": 10000, "test": 5000}[self.split],
                "mnistm": {"train": 5000, "val": 5000, "trainval": 10000, "test": 5000}[self.split],
                #"mnist": {"train": 50000, "val": 10000, "trainval": 60000, "test": 10000}[self.split],
                #"mnistm": {"train": 50000, "val": 10000, "trainval": 60000, "test": 10000}[self.split],
            }[self.domain],
        )

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img, label = self.imgs[idx], self.labels[idx]
        img = np.transpose(img, (1, 2, 0))
        image = self.to_pil(img).convert('RGB')
        if self.transform is not None:
            image = self.transform(image).to(torch.float32)
        label = torch.tensor(label).to(torch.float32)
        return image, label
