from ..layers import CORALLoss
from ..utils import common_functions as c_f
from .base_validator import BaseValidator


class CORALValidator(BaseValidator):
    def __init__(self, layer="features", **kwargs):
        super().__init__(**kwargs)
        self.layer = layer
        self.loss_fn = CORALLoss()

    def compute_score(self, src_train, target_train):
        x = src_train[self.layer]
        y = target_train[self.layer]
        return -self.loss_fn(x, y).item()
