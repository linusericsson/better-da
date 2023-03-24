from torch.nn.functional import mse_loss

from .base_validator import BaseValidator


class MSEValidator(BaseValidator):
    """
    Returns mse using the
    [torchmetrics mean_squared_error function](https://torchmetrics.readthedocs.io/en/latest/regression/mean_squared_error.html#functional-interface).

    The required dataset splits are ```["src_val"]```.
    This can be changed using [```key_map```][pytorch_adapt.validators.BaseValidator.__init__].
    """

    def __init__(self, layer="logits"):
        self.layer = layer
        super().__init__()

    def compute_score(self, src_val):
        return -mse_loss(src_val[self.layer], src_val["labels"], reduction="mean").item()
