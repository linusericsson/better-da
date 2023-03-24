import torch
import torch.nn as nn


class Classifier(nn.Module):
    """
    A 3-layer MLP for classification.
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
        self.linear1 = nn.Linear(in_size, h)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout()
        self.linear2 = nn.Linear(h, h // 2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout()
        self.linear3 = nn.Linear(h // 2, num_classes)

    def forward(self, x, return_all_features=False):
        """"""
        fl3 = self.dropout1(self.relu1(self.linear1(x)))
        fl6 = self.dropout2(self.relu2(self.linear2(fl3)))
        if return_all_features:
            return self.linear3(fl6), fl6, fl3
        else:
            return self.linear3(fl6)


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
