import torch
import torch.nn as nn


class Regressor(nn.Module):
    """
    A 3-layer MLP for regression.
    """

    def __init__(self, num_outputs, max_output, in_size=2048, h=256):
        """
        Arguments:
            num_classes: size of the output
            in_size: size of the input
            h: hidden layer size
        """
        super().__init__()
        self.max_output = max_output
        self.h = h
        self.linear1 = nn.Linear(in_size, h)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout()
        self.linear2 = nn.Linear(h, h // 2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout()
        self.linear3 = nn.Linear(h // 2, num_outputs)

    def forward(self, x, return_all_features=False):
        """"""
        fl3 = self.dropout1(self.relu1(self.linear1(x)))
        fl6 = self.dropout2(self.relu2(self.linear2(fl3)))
        l3 = self.linear3(fl6)
        if return_all_features:
            return torch.sigmoid(l3) * self.max_output, fl6, fl3
        else:
            return torch.sigmoid(l3) * self.max_output
