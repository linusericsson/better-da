import os
import pickle as pkl
from os.path import join

import numpy as np
from PIL import Image

import torch
from torchvision import transforms

from .base_dataset import BaseDownloadableDataset
from .utils import check_length, check_split

from .cityscapes import Cityscapes
from .gta5 import GTA5
from .synthia import SYNTHIA


class SYNTHIACityscapes():
    """
    Segmentation from SYNTHIA to Cityscapes
    """

    def __init__(self, root: str, domain: str, split: str, transform=None, base_size=512, crop_size=512, **kwargs):
        """
        Arguments:
            root: The dataset must be located at ```<root>/```
            domain: one of synthia/cityscapes_synthia
            split: Which data split to use (train/val/trainval/test).
            transform: The image transform applied to each sample.
        """
        self.root = root
        self.domain = domain
        if self.domain == "synthia":
            split = {"trainval": "train", "test": "val"}[split] # undoing the definintions in get_three_split_datasets (getters.py)
        self.split = check_split(split)

        if self.domain == "synthia":
            dataset_class = SYNTHIA
            data_root = join(root, "SYNTHIA-RAND-CITYSCAPES")
        elif self.domain == "cityscapes":
            dataset_class = Cityscapes
            data_root = join(root, "cityscapes")

        if self.split in ["train", "trainval"]:
            training = True
        else:
            training = False

        self.dataset = dataset_class(
            root=data_root,
            split=self.split,
            base_size=base_size,
            crop_size=crop_size,
            training=training,
            class_16=True
        )

        check_length(
            self.dataset,
            {
                "synthia":  {"train": 9300, "val": 100, "trainval": None, "test": None}[self.split],
                "cityscapes": {"train": 2475, "val": 500, "trainval": 2975, "test": 500}[self.split],
            }[self.domain],
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label_img = self.dataset[idx]
        return img, label_img.long()


class GTA5Cityscapes():
    """
    Segmentation from GTA5 to Cityscapes
    """

    def __init__(self, root: str, domain: str, split: str, transform=None, base_size=64, crop_size=64, **kwargs):
        """
        Arguments:
            root: The dataset must be located at ```<root>/```
            domain: one of gta5/citysapes
            split: Which data split to use (train/val/trainval/test).
            transform: The image transform applied to each sample.
        """
        self.root = root
        self.domain = domain
        if self.domain == "gta5":
            split = {"trainval": "train", "test": "val"}[split] # undoing the definintions in get_three_split_datasets (getters.py)
        self.split = check_split(split)

        if self.domain == "gta5":
            dataset_class = GTA5
            data_root = join(root, "gta5")
        elif self.domain == "cityscapes":
            dataset_class = Cityscapes
            data_root = join(root, "cityscapes")

        if self.split in ["train", "trainval"]:
            training = True
        else:
            training = False

        self.dataset = dataset_class(
            root=data_root,
            split=self.split,
            base_size=base_size,
            crop_size=crop_size,
            training=training,
            class_16=False
        )

        check_length(
            self.dataset,
            {
                "gta5":  {"train": 23966, "val": 508, "trainval": 24474, "test": None}[self.split],
                "cityscapes": {"train": 2475, "val": 500, "trainval": 2975, "test": 500}[self.split],
            }[self.domain],
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label_img = self.dataset[idx]
        return img, label_img.long()
