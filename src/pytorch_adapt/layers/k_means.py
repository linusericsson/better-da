from typing import Tuple

import numpy as np

import torch
import torch.nn.functional as F
from pytorch_metric_learning.utils import common_functions as pml_cf

from sklearn.cluster import KMeans as sklearn_k_means

from ..utils import common_functions as c_f


def check_centroid_init(centroid_init):
    if centroid_init not in ["label_centers", None]:
        raise ValueError("centroid_init should be 'label_centers' or None")


class KMeans(torch.nn.Module):
    """
    Implementation of the pseudo labeling step in
    [Domain Adaptation with Auxiliary Target Domain-Oriented Classifier](https://arxiv.org/abs/2007.04171).
    """

    def __init__(
        self,
        num_classes: int,
        centroid_init=None,
        feat_normalizer=None,
    ):
        """
        Arguments:
            num_classes: The number of class labels in the target dataset.
        """

        super().__init__()
        self.num_classes = num_classes
        check_centroid_init(centroid_init)
        self.centroid_init = centroid_init
        self.feat_normalizer = feat_normalizer

    def forward(
        self,
        features: torch.Tensor,
        logits: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Arguments:
            features: The features to compute pseudolabels for.
            logits: The logits from which predictions will be computed and
                stored in memory. Required if ```update = True```
        """
        with torch.no_grad():
            if self.feat_normalizer:
                features = self.feat_normalizer(features)
            labels = torch.argmax(logits, dim=1)
            pseudo_labels = get_cluster_labels(features, labels, self.num_classes, self.centroid_init)
        return pseudo_labels



# copied from https://github.com/lr94/abas/blob/master/model_selection.py
def get_centroids(data, labels, num_classes):
    centroids = np.zeros((num_classes, data.shape[1]))
    for cid in range(num_classes):
        # Since we are using pseudolabels to compute centroids, some classes might not have instances according to the
        # pseudolabels assigned by the current model. In that case .mean() would return NaN causing KMeans to fail.
        # We set to 0 the missing centroids
        if (labels == cid).any():
            centroids[cid] = data[labels == cid].mean(0)

    return centroids


# adapted from https://github.com/lr94/abas/blob/master/model_selection.py
def get_cluster_labels(features, labels, num_classes, centroid_init):
    features, labels = features.cpu().numpy(), labels.cpu().numpy()

    if centroid_init == "label_centers":
        centroids = get_centroids(features, labels, num_classes)
        clustering = sklearn_k_means(n_clusters=num_classes, init=centroids, n_init=1) # replace with kmeans_pytorch (https://github.com/subhadarship/kmeans_pytorch)
    elif centroid_init is None:
        clustering = sklearn_k_means(n_clusters=num_classes) # replace with kmeans_pytorch (https://github.com/subhadarship/kmeans_pytorch)

    clustering.fit(features)

    return torch.from_numpy(clustering.labels_).cuda().long()
