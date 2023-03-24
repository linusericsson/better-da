import torch

from ..layers import BNMLoss
from .simple_loss_validator import SimpleLossValidator


class BNMValidator(SimpleLossValidator):
    """
    Returns the negative of the
    [BNM loss][pytorch_adapt.layers.bnm_loss.BNMLoss]
    of all logits.
    """

    @property
    def loss_fn(self):
        return BNMLoss()


class BNMWithSourceValidator(BNMValidator):
    """
    Returns the negative of the
    [BNM loss][pytorch_adapt.layers.bnm_loss.BNMLoss]
    of all logits from both source_train and target_train.
    """

    def compute_score(self, src_train, target_train):
        feats = torch.cat([src_train[self.layer], target_train[self.layer]])
        return -self.loss_fn(feats).item()
