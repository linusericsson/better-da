import os

from torchvision import datasets as torch_datasets

from .base_dataset import BaseDataset, BaseDownloadableDataset
from .utils import check_img_paths, check_length, check_split


class OfficeHomeFull(BaseDataset):
    """
    A dataset consisting of 65 classes in 4 domains:
    art, clipart, product, and real.
    """

    def __init__(self, root: str, domain: str, transform):
        """
        Arguments:
            root: The dataset must be located at ```<root>/officehome```
            domain: One of ```"art", "clipart", "product", "real"```.
            transform: The image transform applied to each sample.
        """
        super().__init__(domain=domain)
        self.transform = transform
        self.dataset = torch_datasets.ImageFolder(
            os.path.join(root, "officehome", domain),
            transform=self.transform,
        )
        check_length(
            self, {"art": 2427, "clipart": 4365, "product": 4439, "real": 4357}[domain]
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class OfficeHome(BaseDownloadableDataset):
    """
    A custom train/test split of [OfficeHomeFull][pytorch_adapt.datasets.OfficeHomeFull].

    Extends [BaseDownloadableDataset][pytorch_adapt.datasets.BaseDownloadableDataset],
    so the dataset can be downloaded by setting ```download=True``` when
    initializing.
    """

    url = "https://cornell.box.com/shared/static/xwsbubtcr8flqfuds5f6okqbr3z0w82t"
    filename = "officehome_resized.tar.gz"
    md5 = "52d6039512434aa561d66de9c10828c3"

    def __init__(self, root: str, domain: str, split: str, transform=None, **kwargs):
        """
        Arguments:
            root: The dataset must be located at ```<root>/officehome```
            domain: One of ```"art", "clipart", "product", "real"```.
            train: Whether or not to use the training set.
            transform: The image transform applied to each sample.
        """
        self.split = check_split(split)
        super().__init__(root=root, domain=domain, **kwargs)
        self.transform = transform

    def set_paths_and_labels(self, root):
        labels_file = os.path.join(root, "officehome", f"{self.domain}_{self.split}.txt")
        img_dir = os.path.join(root, "officehome")

        with open(labels_file) as f:
            content = [line.rstrip().split(" ") for line in f]
        self.img_paths = [os.path.join(img_dir, x[0]) for x in content]
        check_img_paths(img_dir, self.img_paths, self.domain)
        check_length(
            self,
            {
                "art": {"train": 1456, "val": 485, "trainval": 1941, "test": 486}[self.split],
                "clipart": {"train": 2619, "val": 873, "trainval": 3492, "test": 873}[self.split],
                "product": {"train": 2663, "val": 887, "trainval": 3550, "test": 889}[self.split],
                "real": {"train": 2614, "val": 871, "trainval": 3485, "test": 872}[self.split],
            }[self.domain],
        )
        self.labels = [int(x[1]) for x in content]
