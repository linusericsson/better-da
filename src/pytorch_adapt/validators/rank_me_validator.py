import torch
from torch.linalg import svdvals

from .base_validator import BaseValidator


class RankMeValidator(BaseValidator):
    def __init__(
        self,
        layer="features",
        with_target=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.layer = layer
        self.with_target = with_target

    def _required_data(self):
        x = ["src_train"]
        if self.with_target:
            x.append("target_train")
        return x

    def rank_me(self, Z, eps=1e-7):
        N = min(Z.shape)
        s = svdvals(Z)
        p = (s / s.abs().sum()) + eps
        p = p[:N]
        return (-(p * p.log()).sum()).exp().item()

    def compute_score(self, src_train, target_train=None):
        features = src_train[self.layer]

        if self.with_target:
            tgt_features = target_train[self.layer]
            features = torch.cat([features, tgt_features])

        return self.rank_me(features)
