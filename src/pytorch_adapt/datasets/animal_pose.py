import os, json, shutil

from PIL import Image

import torch
from torchvision import datasets as torch_datasets
import torchvision.transforms.functional as TF

from .base_dataset import BaseDataset, BaseDownloadableDataset
from .utils import check_img_paths, check_length, check_split


class AnimalPoseFull(BaseDataset):
    """
    Animal pose estimation dataset with two domains, synthetic and real.
    """

    def __init__(self, root: str, domain: str, transform):
        """
        Arguments:
            root: The dataset must be located at ```<root>/AnimalPose```
            domain: One of ```"synthetic", "real"```.
            transform: The image transform applied to each sample.
        """

        super().__init__(domain=domain)
        self.transform = transform
        self.dataset = torch_datasets.ImageFolder(
            os.path.join(root, "AnimalPose", domain), transform=self.transform
        )
        check_length(self, {"synthetic": 20000, "real": 3237+2038+842}[domain])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class AnimalPose(BaseDownloadableDataset):
    """
    A custom train/test split of [AnimalPoseFull][pytorch_adapt.datasets.AnimalPoseFull].

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
            root: The dataset must be located at ```<root>/AnimalPose```
            domain: One of ```"synthetic", "real"```.
            train: Whether or not to use the training set.
            transform: The image transform applied to each sample.
        """
        self.split = check_split(split)
        super().__init__(root=root, domain=domain, **kwargs)
        self.transform = transform
        self.crop_size = 224
        self.resize = 224

    def set_paths_and_labels(self, root):
        if self.domain == "synthetic":
            if self.split == "trainval":
                info_file = json.load(open(os.path.join(root, "AnimalPose", "synthetic", "train", "info.json"), "r"))
                info_file2 = json.load(open(os.path.join(root, "AnimalPose", "synthetic", "val", "info.json"), "r"))
                info_file["file_name"] += info_file2["file_name"]
                info_file["regression_label"] += info_file2["regression_label"]
                img_dir = os.path.join(root, "AnimalPose", "synthetic", "images")
            else:
                info_file = json.load(open(os.path.join(root, "AnimalPose", "synthetic", self.split, "info.json"), "r"))
                img_dir = os.path.join(root, "AnimalPose", "synthetic", "images")
        elif self.domain == "real":
            if self.split == "trainval":
                info_file = json.load(open(os.path.join(root, "AnimalPose", "real", "train", "info.json"), "r"))
                info_file2 = json.load(open(os.path.join(root, "AnimalPose", "real", "val", "info.json"), "r"))
                info_file["file_name"] += info_file2["file_name"]
                info_file["regression_label"] += info_file2["regression_label"]
                img_dir = os.path.join(root, "AnimalPose", "real", "images")
            else:
                info_file = json.load(open(os.path.join(root, "AnimalPose", "real", self.split, "info.json"), "r"))
                img_dir = os.path.join(root, "AnimalPose", "real", self.split, "images")

        self.img_paths = [os.path.join(img_dir, x) for x in info_file["file_name"]]
        self.labels = info_file["regression_label"]

        """
        if self.domain == "synthetic":
            n = len(self.img_paths)
            torch.manual_seed(0)
            idx = torch.randperm(n)[:5000]
            print(n, idx.max())

            os.makedirs(os.path.join(root, "AnimalPose", self.domain, "train", "images"), exist_ok=True)
            train_info = {}
            train_info["file_name"] = [info_file["file_name"][i] for i in idx]
            train_info["category"] = [info_file["category"][i] for i in idx]
            train_info["regression_label"] = [info_file["regression_label"][i] for i in idx]
            json.dump(train_info, open(os.path.join(root, "AnimalPose", self.domain, "train", "info.json"), "w"))
            src = os.path.join(root, "AnimalPose", self.domain, "images")
            dst = os.path.join(root, "AnimalPose", self.domain, "train", "images")
            for img_path in train_info["file_name"]:
                shutil.copy(os.path.join(src, img_path), os.path.join(dst, img_path))

            idx = torch.randperm(n)[5000:10000]
            os.makedirs(os.path.join(root, "AnimalPose", self.domain, "val", "images"), exist_ok=True)
            val_info = {}
            val_info["file_name"] = [info_file["file_name"][i] for i in idx]
            val_info["category"] = [info_file["category"][i] for i in idx]
            val_info["regression_label"] = [info_file["regression_label"][i] for i in idx]
            json.dump(val_info, open(os.path.join(root, "AnimalPose", self.domain, "val", "info.json"), "w"))
            src = os.path.join(root, "AnimalPose", self.domain, "images")
            dst = os.path.join(root, "AnimalPose", self.domain, "val", "images")
            for img_path in val_info["file_name"]:
                shutil.copy(os.path.join(src, img_path), os.path.join(dst, img_path))

            idx = torch.randperm(n)[10000:15000]
            os.makedirs(os.path.join(root, "AnimalPose", self.domain, "test", "images"), exist_ok=True)
            test_info = {}
            test_info["file_name"] = [info_file["file_name"][i] for i in idx]
            test_info["category"] = [info_file["category"][i] for i in idx]
            test_info["regression_label"] = [info_file["regression_label"][i] for i in idx]
            json.dump(test_info, open(os.path.join(root, "AnimalPose", self.domain, "test", "info.json"), "w"))
            src = os.path.join(root, "AnimalPose", self.domain, "images")
            dst = os.path.join(root, "AnimalPose", self.domain, "test", "images")
            for img_path in test_info["file_name"]:
                shutil.copy(os.path.join(src, img_path), os.path.join(dst, img_path))
        """

        #check_img_paths(img_dir, self.img_paths, self.domain)
        check_length(
            self,
            {
                "synthetic": {"train": 5000, "val": 5000, "trainval": 10000, "test": 5000}[self.split],
                "real": {"train": 3237, "val": 2038, "trainval": 3237+2038, "test": 842}[self.split],
            }[self.domain],
        )

    def __getitem__(self, idx):
        image_path, pose = self.img_paths[idx], self.labels[idx]

        image = Image.open(image_path).convert("RGB")
        width, height = image.width, image.height
        if self.transform is not None:
            image = self.transform(image)
            new_height, new_width = image.shape[-2:]
        else:
            new_width, new_height = image.width, image.height

        pose = torch.tensor(pose).flatten()
        pose = self.update_pose_after_crop(pose, width, height, new_width, new_height)

        return image, pose

    def update_pose_after_crop(self, pose, old_width, old_height, new_width, new_height):
        # first deal with resizing
        resizing_scale = min(new_width, new_height) / min(old_width, old_height)
        pose = [x * resizing_scale for x in pose]
        # second deal with crop
        x_diff, y_diff = ((old_width  * resizing_scale) - new_width) / 2., ((old_height * resizing_scale) - new_height) / 2.
        pose[0::2] = [max(x - x_diff,  0) for x in pose[0::2]]
        pose[1::2] = [max(y - y_diff, 0) for y in pose[1::2]]
        pose[0::2] = [0 if x < 0 or x > new_width else x for x in pose[0::2]]
        pose[1::2] = [0 if y < 0 or y > new_height or x == 0 else y for x, y in zip(pose[0::2], pose[1::2])]
        pose = torch.tensor(pose).float()
        return pose