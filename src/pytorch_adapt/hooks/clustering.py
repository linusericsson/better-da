from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F

from ..layers import KMeans
from ..utils import common_functions as c_f
from .base import BaseHook
from .features import FeaturesAndLogitsHook


class ClusteringHook(BaseHook):
    """
    Creates pseudo labels for the target domain
    using k-nearest neighbors. Then computes a
    classification loss based on these pseudo labels.
    """

    def __init__(
        self, num_classes, loss_fn=None, with_src=False, centroid_init=None, feat_normalize=False, **kwargs
    ):
        """
        Arguments:
            dataset_size: The number of samples in the target dataset.
            feature_dim: The feature dimensionality, i.e at each iteration
                the features should be size ```(N, D)``` where N is batch size and
                D is ```feature_dim```.
            num_classes: The number of class labels in the target dataset.
            loss_fn: The classification loss function.
                If ```None``` it defaults to
                ```torch.nn.CrossEntropyLoss```.
        """
        super().__init__(**kwargs)
        self.loss_fn = c_f.default(
            loss_fn, torch.nn.CrossEntropyLoss, {"reduction": "none"}
        )
        self.with_src = with_src
        centroid_init = "label_centers" if centroid_init else None
        feat_normalizer = F.normalize if feat_normalize else None
        self.hook = FeaturesAndLogitsHook(domains=["target"])
        self.labeler = KMeans(num_classes, centroid_init=centroid_init, feat_normalizer=feat_normalizer)

    def call(self, inputs, losses) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        outputs = self.hook(inputs, losses)[0]
        if self.with_src:
            keys = list(outputs.keys()) + list(inputs.keys())
            [src_features, target_features, src_logits, target_logits] = c_f.extract(
                [inputs, outputs],
                c_f.filter(keys, "", ["_features$", "_logits$"]),
            )
            features, logits = torch.cat([src_features, target_features], dim=0), torch.cat([src_logits, target_logits], dim=0)
        else:
            [features, logits] = c_f.extract(
                [inputs, outputs],
                c_f.filter(self.hook.out_keys, "", ["_features$", "_logits$"]),
            )
        pseudo_labels = self.labeler(features, logits)
        loss = self.loss_fn(logits, pseudo_labels).mean()
        return outputs, {"pseudo_label_loss": loss}

    def _loss_keys(self):
        """"""
        return ["pseudo_label_loss"]

    def _out_keys(self):
        """"""
        return self.hook.out_keys
