import torch

from .diversity_validator import DiversityValidator
from .entropy_validator import EntropyValidator
from .multiple_validators import MultipleValidators


class IMValidator(MultipleValidators):
    """
    The sum of [EntropyValidator][pytorch_adapt.validators.EntropyValidator]
    and [DiversityValidator][pytorch_adapt.validators.DiversityValidator]
    """

    def __init__(self, weights=None, **kwargs):
        self.layer = kwargs.pop("layer", None)
        inner_kwargs = {} if not self.layer else {"layer": self.layer}
        validators = {
            "entropy": EntropyValidator(**inner_kwargs),
            "diversity": DiversityValidator(**inner_kwargs),
        }
        super().__init__(validators=validators, weights=weights, **kwargs)


class IMCombinedValidator(IMValidator):
    """
    The sum of [EntropyValidator][pytorch_adapt.validators.EntropyValidator]
    and [DiversityValidator][pytorch_adapt.validators.DiversityValidator]
    """

    def __init__(self, weights=None, **kwargs):
        super().__init__(weights=weights, **kwargs)

    def __call__(self, src_train, target_train):
        if self.layer is None:
            combined = {"logits": torch.cat([src_train["logits"], target_train["logits"]])}
        else:
            combined = {self.layer: torch.cat([src_train[self.layer], target_train[self.layer]])}
        return super().__call__(target_train=combined)
