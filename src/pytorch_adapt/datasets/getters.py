from torchvision.datasets import MNIST

from ..transforms.classification import get_mnist_transform, get_resnet_transform
from ..transforms.regression import get_resnet_regression_transform
from ..transforms.segmentation import (
    get_mnist_segmentation_transform, get_mnist_segmentation_label_transform,
    get_resnet_segmentation_transform, get_resnet_segmentation_label_transform,
)
from ..utils import common_functions as c_f
from .clipart1k import Clipart1kMultiLabel
from .combined_source_and_target import CombinedSourceAndTargetDataset
from .concat_dataset import ConcatDataset
from .domainnet import DomainNet126
from .mnistm import MNISTM, BaseMNIST
from .mnistmr import MNISTMR
from .mnistms import MNISTMS
from .office31 import Office31
from .officehome import OfficeHome
from .dogs_and_birds import DogsAndBirds
from .animal_pose import AnimalPose
from .source_dataset import SourceDataset
from .target_dataset import TargetDataset
from .voc_multilabel import VOCMultiLabel
from .voc_multilabel import get_labels_as_vector as voc_labels_as_vector
from .visda2017_classification import VisDA2017Classification
from .biwi import BIWI
from .faces300w import Faces300W
from .urban_scenes import SYNTHIACityscapes, GTA5Cityscapes


def get_multiple(dataset_getter, domains, **kwargs):
    return ConcatDataset([dataset_getter(domain=d, **kwargs) for d in domains])


def get_datasets(
    dataset_getter,
    src_domains,
    target_domains,
    folder,
    download=False,
    return_target_with_labels=False,
    supervised=False,
    transform_getter=None,
    **kwargs,
):
    def getter(domains, train, is_training):
        return get_multiple(
            dataset_getter,
            domains,
            train=train,
            is_training=is_training,
            root=folder,
            download=download,
            transform_getter=transform_getter,
            **kwargs,
        )

    if not src_domains and not target_domains:
        raise ValueError(
            "At least one of src_domains and target_domains must be provided"
        )

    output = {}
    if src_domains:
        output["src_train"] = SourceDataset(getter(src_domains, True, False))
        output["src_val"] = SourceDataset(getter(src_domains, False, False))
    if target_domains:
        output["target_train"] = TargetDataset(
            getter(target_domains, True, False), supervised=supervised
        )
        output["target_val"] = TargetDataset(
            getter(target_domains, False, False), supervised=supervised
        )
        # For academic setting: unsupervised learning w/ seperate target datasets that have gt lables for eval.
        if return_target_with_labels:
            output["target_train_with_labels"] = TargetDataset(
                getter(target_domains, True, False), domain=1, supervised=True
            )
            output["target_val_with_labels"] = TargetDataset(
                getter(target_domains, False, False), domain=1, supervised=True
            )
    if src_domains and target_domains:
        output["train"] = CombinedSourceAndTargetDataset(
            SourceDataset(getter(src_domains, True, True)),
            TargetDataset(getter(target_domains, True, True)),
        )
    elif src_domains:
        output["train"] = SourceDataset(getter(src_domains, True, True))
    elif target_domains:
        output["train"] = TargetDataset(getter(target_domains, True, True))
    return output


def get_three_split_datasets(
    dataset_getter,
    src_domains,
    target_domains,
    folder,
    download=False,
    return_target_with_labels=False,
    supervised=False,
    transform_getter=None,
    target_transform_getter=None,
    **kwargs,
):
    def getter(domains, split, is_training):
        return get_multiple(
            dataset_getter,
            domains,
            split=split,
            is_training=is_training,
            root=folder,
            download=download,
            transform_getter=transform_getter,
            target_transform_getter=target_transform_getter,
            **kwargs,
        )

    if not src_domains and not target_domains:
        raise ValueError(
            "At least one of src_domains and target_domains must be provided"
        )

    output = {}
    if src_domains:
        output["src_train"] = SourceDataset(getter(src_domains, "trainval", False))
        output["src_val"] = SourceDataset(getter(src_domains, "test", False))
    if target_domains:
        output["target_train"] = TargetDataset(
            getter(target_domains, "train", False), supervised=supervised
        )
        output["target_val"] = TargetDataset(
            getter(target_domains, "val", False), supervised=supervised
        )
        output["target_test"] = TargetDataset(
            getter(target_domains, "test", False), supervised=supervised
        )
        # For academic setting: unsupervised learning w/ seperate target datasets that have gt lables for eval.
        if return_target_with_labels:
            output["target_train_with_labels"] = TargetDataset(
                getter(target_domains, "train", False), domain=1, supervised=True
            )
            output["target_val_with_labels"] = TargetDataset(
                getter(target_domains, "val", False), domain=1, supervised=True
            )
            output["target_test_with_labels"] = TargetDataset(
                getter(target_domains, "test", False), domain=1, supervised=True
            )
    if src_domains and target_domains:
        output["train"] = CombinedSourceAndTargetDataset(
            SourceDataset(getter(src_domains, "trainval", True)),
            TargetDataset(getter(target_domains, "train", True)),
        )
    elif src_domains:
        output["train"] = SourceDataset(getter(src_domains, "trainval", True))
    elif target_domains:
        output["train"] = TargetDataset(getter(target_domains, "train", True))
    return output


def _get_mnist_mnistm(is_training, transform_getter, **kwargs):
    transform_getter = c_f.default(transform_getter, get_mnist_transform)
    domain = kwargs["domain"]
    kwargs["transform"] = transform_getter(
        domain=domain, train=kwargs["train"], is_training=is_training
    )
    kwargs.pop("domain")
    if domain == "mnist":
        return MNIST(**kwargs)
    elif domain == "mnistm":
        return MNISTM(**kwargs)


def get_mnist_mnistm(*args, **kwargs):
    return get_datasets(_get_mnist_mnistm, *args, **kwargs)


def standard_dataset(cls):
    def fn(is_training, transform_getter, **kwargs):
        transform_getter = c_f.default(transform_getter, get_resnet_transform)
        kwargs["transform"] = transform_getter(
            domain=kwargs["domain"], train=kwargs["train"], is_training=is_training
        )
        return cls(**kwargs)

    return fn


def three_split_dataset(cls):
    def fn(is_training, transform_getter, **kwargs):
        transform_getter = c_f.default(transform_getter, get_resnet_transform) if "mnist" not in kwargs["domain"] else get_mnist_transform
        kwargs["transform"] = transform_getter(
            domain=kwargs["domain"], train=(kwargs["split"] in ["train", "trainval"]), is_training=is_training
        )
        return cls(**kwargs)

    return fn

def three_split_regression_dataset(cls):
    def fn(is_training, transform_getter, **kwargs):
        transform_getter = c_f.default(transform_getter, get_resnet_regression_transform) if "mnist" not in kwargs["domain"] else get_mnist_transform
        kwargs["transform"] = transform_getter(
            domain=kwargs["domain"], train=(kwargs["split"] in ["train", "trainval"]), is_training=is_training
        )
        return cls(**kwargs)

    return fn

def three_split_segmentation_dataset(cls):
    def fn(is_training, transform_getter, target_transform_getter, **kwargs):
        transform_getter = c_f.default(transform_getter, get_resnet_segmentation_transform) if "mnist" not in kwargs["domain"] else get_mnist_segmentation_transform
        target_transform_getter = c_f.default(target_transform_getter, get_resnet_segmentation_label_transform) if "mnist" not in kwargs["domain"] else get_mnist_segmentation_label_transform
        kwargs["transform"] = transform_getter(
            domain=kwargs["domain"], train=(kwargs["split"] in ["train", "trainval"]), is_training=is_training
        )
        kwargs["target_transform"] = target_transform_getter(
            domain=kwargs["domain"], train=(kwargs["split"] in ["train", "trainval"]), is_training=is_training
        )
        return cls(**kwargs)

    return fn


def get_office31(*args, **kwargs):
    return get_three_split_datasets(three_split_dataset(Office31), *args, **kwargs)

def get_mnist(*args, **kwargs):
    output = get_three_split_datasets(three_split_dataset(BaseMNIST), *args, **kwargs)
    [output.pop(k) for k in list(output.keys()) if "target" in k]
    output.pop("train")

    output2 = get_three_split_datasets(three_split_dataset(MNISTM), *args, **kwargs)
    [output2.pop(k) for k in list(output2.keys()) if "src" in k]
    output2.pop("train")

    output["train"] = CombinedSourceAndTargetDataset(output["src_train"], output2["target_train_with_labels"])
    output.update(output2)

    return output


def get_officehome(*args, **kwargs):
    return get_three_split_datasets(three_split_dataset(OfficeHome), *args, **kwargs)


def get_domainnet126(*args, **kwargs):
    return get_datasets(standard_dataset(DomainNet126), *args, **kwargs)

def get_visda2017c(*args, **kwargs):
    return get_three_split_datasets(three_split_dataset(VisDA2017Classification), *args, **kwargs)

def get_biwi(*args, **kwargs):
    return get_three_split_datasets(three_split_regression_dataset(BIWI), *args, **kwargs)

def get_mnistmr(*args, **kwargs):
    return get_three_split_datasets(three_split_regression_dataset(MNISTMR), *args, **kwargs)

def get_mnistms(*args, **kwargs):
    return get_three_split_datasets(three_split_segmentation_dataset(MNISTMS), *args, **kwargs)

def get_synthia_cityscapes(*args, **kwargs):
    return get_three_split_datasets(three_split_segmentation_dataset(SYNTHIACityscapes), *args, **kwargs)

def get_gta5_cityscapes(*args, **kwargs):
    return get_three_split_datasets(three_split_segmentation_dataset(GTA5Cityscapes), *args, **kwargs)

def get_mnistms(*args, **kwargs):
    return get_three_split_datasets(three_split_segmentation_dataset(MNISTMS), *args, **kwargs)

def get_dogs_and_birds(*args, **kwargs):
    return get_three_split_datasets(three_split_regression_dataset(DogsAndBirds), *args, **kwargs)

def get_animal_pose(*args, **kwargs):
    return get_three_split_datasets(three_split_regression_dataset(AnimalPose), *args, **kwargs)

def get_faces300w(*args, **kwargs):
    return get_three_split_datasets(three_split_regression_dataset(Faces300W), *args, **kwargs)


def _get_voc_multilabel(is_training, transform_getter, **kwargs):
    # import here, because albumentations is an optional dependency
    from ..transforms.detection import VOCTransformWrapper, get_voc_transform

    transform_getter = c_f.default(transform_getter, get_voc_transform)
    domain = kwargs["domain"]
    transform = transform_getter(
        domain=domain, train=kwargs["train"], is_training=is_training
    )
    kwargs["transforms"] = VOCTransformWrapper(transform, voc_labels_as_vector)
    kwargs.pop("domain")
    train = kwargs.pop("train")
    if domain == "voc":
        kwargs["image_set"] = "train" if train else "val"
        return VOCMultiLabel(**kwargs)
    elif domain == "clipart":
        kwargs.pop("year", None)
        kwargs["image_set"] = "train" if train else "test"
        return Clipart1kMultiLabel(**kwargs)


def get_voc_multilabel(*args, **kwargs):
    return get_datasets(_get_voc_multilabel, *args, **kwargs)
