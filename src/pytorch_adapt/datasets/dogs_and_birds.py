import os
import pickle as pkl

import numpy as np
from PIL import Image

import torch
from torchvision import transforms

from .base_dataset import BaseDownloadableDataset
from .utils import check_length, check_split


class DogsAndBirds():
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

        #self.domain_info = {"dogs": ("StanfordDogs", "StanfordDogs/Images"), "birds": ("CUB",  "CUB/CUB_200_2011/images")}
        self.domain_info = {"dogs": ("StanfordDogs", "StanfordDogs/Images"), "birds": ("NABirds",  "NABirds/nabirds/images")}
        data = pkl.load(open(os.path.join(root, f"{self.domain_info[self.domain][0]}/dataset_dict.pkl"), "rb"))
        self.image_folder = os.path.join(root, self.domain_info[self.domain][1])
        self.img_paths = data[self.split]
        self.labels = data[f"{self.split}labels"]
        check_length(
            self,
            {
                #"dogs": {"train": 6418, "val": 6459, "trainval": 12877, "test": 9249}[self.split],
                #"birds": {"train": 2997, "val": 2997, "trainval": 5994, "test": 5794}[self.split],
                "dogs":  {"train": 5000, "val": 5000, "trainval": 10000, "test": 5000}[self.split],
                "birds": {"train": 5000, "val": 5000, "trainval": 10000, "test": 5000}[self.split],
            }[self.domain],
        )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img, label = self.img_paths[idx], self.labels[idx]
        image_path = os.path.join(self.image_folder, img)
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image).to(torch.float32)
        return image, label
