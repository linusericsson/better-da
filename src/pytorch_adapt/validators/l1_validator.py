from ..layers import L1Loss
from .simple_loss_validator import SimpleLossValidator


class L1Validator(SimpleLossValidator):
    """
    Returns 
    [l1][pytorch_adapt.layers.entropy_loss.L1Loss]
    of all logits.
    """

    @property
    def loss_fn(self):
        return L1Loss(after_softmax=self.layer == "preds")
