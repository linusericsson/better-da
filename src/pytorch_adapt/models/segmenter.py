import torch
import torch.nn as nn
import torch.nn.functional as F


class Segmenter(nn.Module):
    """
    A CNN head for segmentation.
    """

    def __init__(self, num_classes, in_size=2048, h=256):
        """
        Arguments:
            num_classes: size of the output
            in_size: size of the input
            h: hidden layer size
        """
        super().__init__()
        self.h = h
        self.conv1 = nn.ConvTranspose2d(48, 32, 4, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.ConvTranspose2d(32, 16, 2, 2)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.ConvTranspose2d(16, num_classes, 2, 2)

    def forward(self, x, return_all_features=False):
        """
        input is [N, 48, 13, 13]
        """
        fl3 = self.relu1(self.bn1(self.conv1(x)))
        fl6 = self.relu2(self.bn2(self.conv2(fl3)))
        if return_all_features:
            return self.conv3(fl6), fl6.flatten(), fl3.flatten()
        else:
            return self.conv3(fl6)


class LinearClassifier(nn.Module):
    """
    A linear layer for classification.
    """

    def __init__(self, num_classes, in_size=2048):
        """
        Arguments:
            num_classes: size of the output
            in_size: size of the input
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_size, num_classes)
        )

    def forward(self, x, return_all_features=False):
        """"""
        if return_all_features:
            return self.net(x), torch.zeros(1), torch.zeros(1)
        else:
            return self.net(x)
