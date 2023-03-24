import os

from PIL import Image

import torch
from torchvision import datasets as torch_datasets
import torchvision.transforms.functional as TF

from .base_dataset import BaseDataset, BaseDownloadableDataset
from .utils import check_img_paths, check_length, check_train, check_split


class BIWIFull(BaseDataset):
    """
    A small dataset consisting of 2 domains:
    female, male.
    """

    def __init__(self, root: str, domain: str, transform):
        """
        Arguments:
            root: The dataset must be located at ```<root>/biwi```
            domain: One of ```"female", "male"```.
            transform: The image transform applied to each sample.
        """

        super().__init__(domain=domain)
        self.transform = transform
        self.dataset = torch_datasets.ImageFolder(
            os.path.join(root, "biwi", domain), transform=self.transform
        )
        check_length(self, {"female": 4698+1176, "male": 7842+1962}[domain])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class BIWI(BaseDownloadableDataset):
    """
    A custom train/test split of [BIWIFull][pytorch_adapt.datasets.BIWIFull].

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
            root: The dataset must be located at ```<root>/biwi```
            domain: One of ```"male", "female"```.
            train: Whether or not to use the training set.
            transform: The image transform applied to each sample.
        """
        self.split = check_split(split)
        super().__init__(root=root, domain=domain, **kwargs)
        self.transform = transform
        self.crop_size = 224
        self.resize = 224

    def set_paths_and_labels(self, root):
        labels_file = os.path.join(root, "biwi", f"{self.domain}_{self.split}.txt")
        img_dir = os.path.join(root, "biwi")

        with open(labels_file) as f:
            content = [line.rstrip().split(" ") for line in f]
        self.img_paths = [os.path.join(img_dir, x[0]) for x in content]
        check_img_paths(img_dir, self.img_paths, self.domain)
        check_length(
            self,
            {
                "female": {"train": 3524, "val": 1174, "trainval": 4698, "test": 1176}[self.split],
                "male": {"train": 5882, "val": 1960, "trainval": 7842, "test": 1962}[self.split],
            }[self.domain],
        )
        self.labels = [os.path.join(img_dir, x[1]) for x in content]

    def __getitem__(self, idx):
        image_path, label_path = self.img_paths[idx], self.labels[idx]

        pose = torch.tensor([list(map(float, line.strip().split(" "))) for line in open(label_path, "r").readlines() if line.strip()])
        label = pose[:3].flatten()
        head_loc = (pose[3][:2] / 2).to(int)
        head_loc = torch.tensor([head_loc[1], head_loc[0]])

        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        centre = (torch.tensor(image.shape[1:]) / 2).to(int)
        resized_image = TF.resized_crop(
            image,
            top=centre[0] + head_loc[0] - int(self.crop_size / 2),
            left=centre[1] + head_loc[1] - int(self.crop_size / 2),
            width=self.crop_size, height=self.crop_size, size=(self.resize, self.resize)
        )
        if self.domain == "male":
            resized_image = TF.gaussian_blur(resized_image, 45)
        return resized_image, label
