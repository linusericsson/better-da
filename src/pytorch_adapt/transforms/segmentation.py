import torch
import torchvision.transforms as T

from .constants import IMAGENET_MEAN, IMAGENET_STD


class GrayscaleToRGB:
    def __call__(self, x):
        if x.size(0) == 3:
            return x
        elif x.size(0) == 1:
            return torch.cat([x, x, x], dim=0)
        else:
            raise Exception("Image is not grayscale (or even RGB).")


def get_mnist_segmentation_transform(domain, **kwargs):
    return T.Compose(
        [
            T.Resize(64),
            T.ToTensor(),
            GrayscaleToRGB(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_mnist_segmentation_label_transform(domain, **kwargs):
    return T.Compose(
        [
            T.Resize(64),
            T.ToTensor(),
        ]
    )


def get_resnet_segmentation_transform(is_training, **kwargs):
    transform = [
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return T.Compose(transform)


def get_resnet_segmentation_label_transform(is_training, **kwargs):
    transform = [
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
    ]
    return T.Compose(transform)


def get_timm_transform(is_training, **kwargs):
    from timm.data.transforms_factory import create_transform

    return create_transform(
        input_size=224, is_training=is_training, auto_augment="original"
    )


def get_simple_transform(is_training, **kwargs):
    transform = [
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return T.Compose(transform)
