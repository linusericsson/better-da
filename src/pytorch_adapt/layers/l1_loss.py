import torch

from ..utils import common_functions as c_f


def l1(logits, other_logits):
    l1 = torch.sum(
        (torch.softmax(logits, dim=1) - torch.softmax(other_logits, dim=1)).abs(), dim=1
    )
    return l1


def l1_after_softmax(preds, other_preds):
    return torch.sum((preds - other_preds).abs())


def get_l1(logits, other_logits, after_softmax):
    if after_softmax:
        return l1_after_softmax(logits, other_logits)
    return l1(logits, other_logits)


class L1Loss(torch.nn.Module):
    """
    Total Variation distance in prediction space
    """

    def __init__(self, after_softmax: bool = False, return_mean: bool = True):
        """
        Arguments:
            after_softmax: If ```True```, then the rows of the input are assumed to
                already have softmax applied to them.
            return_mean: If ```True```, the mean entropy will be returned.
                If ```False```, the entropy per row of the input will be returned.
        """
        super().__init__()
        self.after_softmax = after_softmax
        self.return_mean = return_mean

    def forward(self, logits: torch.Tensor, other_logits: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            logits: Raw logits if ```self.after_softmax``` is False.
                Otherwise each row should be predictions that sum up to 1.
        """
        l1 = get_l1(logits, other_logits, self.after_softmax)
        if self.return_mean:
            return torch.mean(l1)
        return l1

    def extra_repr(self):
        """"""
        return c_f.extra_repr(self, ["after_softmax"])
