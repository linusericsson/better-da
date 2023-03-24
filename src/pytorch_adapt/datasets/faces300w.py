import os
from math import ceil

from PIL import Image

import torch
from torchvision import datasets as torch_datasets
import torchvision.transforms.functional as TF

from .base_dataset import BaseDataset, BaseDownloadableDataset
from .utils import check_img_paths, check_length, check_train, check_split


class Faces300WFull(BaseDataset):
    """
    A small dataset consisting of 2 domains:
    indoor, outdoor.
    """

    def __init__(self, root: str, domain: str, transform):
        """
        Arguments:
            root: The dataset must be located at ```<root>/300W```
            domain: One of ```"indoor", "outdoor"```.
            transform: The image transform applied to each sample.
        """

        super().__init__(domain=domain)
        self.transform = transform
        self.dataset = torch_datasets.ImageFolder(
            os.path.join(root, "300W", domain), transform=self.transform
        )
        check_length(self, {"indoor": 300, "outdoor": 299}[domain])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class Faces300W(BaseDownloadableDataset):
    """
    A custom train/test split of [Faces300WFull][pytorch_adapt.datasets.Faces300WFull].

    Extends [BaseDownloadableDataset][pytorch_adapt.datasets.BaseDownloadableDataset],
    so the dataset can be downloaded by setting ```download=True``` when
    initializing.
    """

    url = ""
    filename = ""
    md5 = ""

    def __init__(self, root: str, domain: str, split: str, transform=None, **kwargs):
        """
        Arguments:
            root: The dataset must be located at ```<root>/300W```
            domain: One of ```"indoor", "outdoor"```.
            train: Whether or not to use the training set.
            transform: The image transform applied to each sample.
        """
        self.split = check_split(split)
        super().__init__(root=root, domain=domain, **kwargs)
        self.transform = transform
        self.padding = 224
        self.resize = 224

    def set_paths_and_labels(self, root):
        labels_file = os.path.join(root, "300W", f"{self.domain}_{self.split}.txt")
        img_dir = os.path.join(root, "300W")

        with open(labels_file) as f:
            content = [line.rstrip().split(" ") for line in f]
        self.img_paths = [os.path.join(img_dir, x[0]) for x in content]
        check_img_paths(img_dir, self.img_paths, self.domain)
        check_length(
            self,
            {
                "indoor": {"train": 180, "val": 60, "trainval": 240, "test": 60}[self.split],
                "outdoor": {"train": 179, "val": 59, "trainval": 238, "test": 61}[self.split],
            }[self.domain],
        )
        self.labels = [os.path.join(img_dir, x[1]) for x in content]

    def __getitem__(self, idx):
        # get image in original resolution
        path = self.img_paths[idx]
        image = Image.open(path).convert("RGB")
        h, w = image.height, image.width
        min_side = min(h, w)

        # get keypoints in original resolution
        keypoint = torch.load(self.labels[idx])
        #bbox_x1, bbox_x2 = keypoint[:, 0].min().item(), keypoint[:, 0].max().item()
        #bbox_y1, bbox_y2 = keypoint[:, 1].min().item(), keypoint[:, 1].max().item()
        #bbox_x1, bbox_x2 = bbox_x1 - self.padding, bbox_x2 + self.padding
        #bbox_y1, bbox_y2 = bbox_y1 - self.padding, bbox_y2 + self.padding
        #bbox_width = ceil(bbox_x2 - bbox_x1)
        #bbox_height = ceil(bbox_y2 - bbox_y1)
        #bbox_length = max(bbox_width, bbox_height)

        #image = TF.resized_crop(image, top=bbox_y1, left=bbox_x1, height=bbox_length, width=bbox_length, size=(self.resize, self.resize))
        image = TF.resize(image, size=(self.resize, self.resize))
        #keypoint = torch.tensor([(x - bbox_x1, y - bbox_y1) for x, y in keypoint])
        
        #h, w = image.height, image.width
        #min_side = min(h, w)

        if self.transform is not None:
            image = self.transform(image)
        #new_h, new_w = image.shape[1:]

        #keypoint = torch.tensor([[
        #    ((x - ((w - min_side) / 2)) / min_side) * new_w,
        #    ((y - ((h - min_side) / 2)) / min_side) * new_h,
        #] for x, y in keypoint])
        keypoint = keypoint.flatten()
        keypoint = keypoint / min_side

        return image, keypoint
