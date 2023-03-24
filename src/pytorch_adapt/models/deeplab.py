from functools import partial
from collections import OrderedDict
from typing import Any, List, Dict, Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from torchvision.transforms._presets import SemanticSegmentation
from torchvision.models._api import Weights, WeightsEnum
from torchvision.models._meta import _VOC_CATEGORIES
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.resnet import resnet50


__all__ = [
    "DeepLabV3",
    "DeepLabV3_ResNet50_Weights",
    "DeepLabV3_ResNet101_Weights",
    "DeepLabV3_MobileNet_V3_Large_Weights",
    "deeplabv3_mobilenet_v3_large",
    "deeplabv3_resnet50",
    "deeplabv3_resnet101",
]


_COMMON_META = {
    "categories": _VOC_CATEGORIES,
    "min_size": (1, 1),
    "_docs": """
        These weights were trained on a subset of COCO, using only the 20 categories that are present in the Pascal VOC
        dataset.
    """,
}


class DeepLabV3_ResNet50_Weights(WeightsEnum):
    COCO_WITH_VOC_LABELS_V1 = Weights(
        url="https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth",
        transforms=partial(SemanticSegmentation, resize_size=520),
        meta={
            **_COMMON_META,
            "num_params": 42004074,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/segmentation#deeplabv3_resnet50",
            "_metrics": {
                "COCO-val2017-VOC-labels": {
                    "miou": 66.4,
                    "pixel_acc": 92.4,
                }
            },
            "_ops": 178.722,
            "_file_size": 160.515,
        },
    )
    DEFAULT = COCO_WITH_VOC_LABELS_V1


class DeepLabV3G(nn.Module):
    def __init__(self, pretrained=True) -> None:
        super().__init__()
        backbone = resnet50(weights="DEFAULT", replace_stride_with_dilation=[False, True, True])
        return_layers = {"layer4": "out"}
        self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
        if pretrained:
            weights = DeepLabV3_ResNet50_Weights.verify("DEFAULT")
            self.load_state_dict(weights.get_state_dict(progress=True), strict=False)

    def forward(self, x: Tensor):
        # contract: features is a dict of tensors
        x = self.backbone(x)["out"]
        return x


class DeepLabV3C(nn.Module):
    def __init__(self, num_classes=17, input_shape=(512, 512)) -> None:
        super().__init__()
        self.num_classes = num_classes
        if self.num_classes == 11: input_shape = (64, 64) # for MNISTMS
        self.classifier = DeepLabHead(2048, self.num_classes)
        self.input_shape = input_shape
        weights = DeepLabV3_ResNet50_Weights.verify("DEFAULT")
        state_dict = weights.get_state_dict(progress=True)
        del state_dict['classifier.4.weight']
        del state_dict['classifier.4.bias']
        self.load_state_dict(state_dict, strict=False)

    def forward(self, x, return_all_features=False):
        fl3 = self.classifier[0](x)
        fl6 = self.classifier[3](self.classifier[2](self.classifier[1](fl3)))
        x = self.classifier[4](fl6)
        x = F.interpolate(x, size=self.input_shape, mode="bilinear", align_corners=False)
        if return_all_features:
            return x, fl6.flatten(start_dim=1), fl3.flatten(start_dim=1)
        else:
            return x


class Lambda(nn.Module):
    def __init__(self, lambd):
        super(Lambda, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)


class DeepLabV3D(nn.Module):
    def __init__(self, input_class="C", num_classes=17, input_shape=(512, 512)) -> None:
        super().__init__()
        self.num_classes = num_classes
        if self.num_classes == 11: input_shape = (64, 64) # for MNISTMS
        if input_class == "C":
            self.discriminator = nn.Sequential(
                torch.nn.Flatten(start_dim=1),
                nn.Linear(num_classes * input_shape[0] * input_shape[1], 1),
                torch.nn.Flatten(start_dim=0)
            )
        elif input_class == "G":
            self.discriminator = nn.Sequential(
                nn.Conv2d(2048, 256, 3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.AvgPool2d(8),
                nn.Flatten(start_dim=1),
                nn.Linear(64, 1),
                nn.Flatten(start_dim=0)
            )

    def forward(self, x):
        x = self.discriminator(x)
        return x


class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__(
            ASPP(in_channels, [12, 24, 36]),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1),
        )


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
        super().__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels: int, atrous_rates: List[int], out_channels: int = 256) -> None:
        super().__init__()
        modules = []
        modules.append(
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())
        )

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        return self.project(res)

if __name__ == "__main__":
    G = DeepLabV3G().cuda()
    C = DeepLabV3C().cuda()
    x = torch.randn(2, 3, 512, 512).cuda()
    x = G(x)
    print(x.shape)
    x = C(x)
    print(x.shape)

    G = DeepLabV3G().cuda()
    C = DeepLabV3C(11).cuda()
    x = torch.randn(2, 3, 64, 64).cuda()
    x = G(x)
    print(x.shape)
    x = C(x)
    print(x.shape)
