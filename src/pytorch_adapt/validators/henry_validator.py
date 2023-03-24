import torch
from torch.nn.functional import mse_loss
from torchmetrics import ConfusionMatrix

from pytorch_adapt.validators.base_validator import BaseValidator

import numpy as np
from statsmodels.nonparametric.kernel_density import KDEMultivariate

from tqdm import tqdm


class HenryValidator(BaseValidator):
    """
    The sum of [ErrorValidator][pytorch_adapt.validators.ErrorValidator]
    and [L1Validator][pytorch_adapt.validators.L1Validator]
    src_val["labels"], src_val["preds"], target_train["preds"]
    """

    def __init__(self, num_classes, layer="preds"):
        self.num_classes = num_classes
        self.layer = layer
        self.confmat = ConfusionMatrix(num_classes=self.num_classes, normalize="true")
        super().__init__()

    def compute_error(self, src_val):
        src_label_preds = torch.softmax(src_val[self.layer], dim=-1).argmax(dim=-1)
        return (src_label_preds != src_val["labels"]).to(torch.float32).mean()

    def compute_distance(self, src_val, target_train):
        self.confmat = self.confmat.to(src_val[self.layer].device)
        src_confmat = self.confmat(src_val[self.layer], src_val["labels"]).to("cpu")

        tgt_label_preds = torch.softmax(target_train[self.layer], dim=-1).argmax(dim=-1).to(torch.float32).to("cpu")
        tgt_histogram = torch.histogram(tgt_label_preds, bins=self.num_classes, density=False).hist / tgt_label_preds.size(0)

        return (src_confmat.T - tgt_histogram).abs().max()

    def compute_score(self, src_val, target_train):
        error = self.compute_error(src_val)
        distance = self.compute_distance(src_val, target_train)
        return 1. - ((error + distance).item() / 2.)


class HenryV3Validator(BaseValidator):
    """
    The sum of [ErrorValidator][pytorch_adapt.validators.ErrorValidator]
    and [L1Validator][pytorch_adapt.validators.L1Validator]
    src_val["labels"], src_val["preds"], target_train["preds"]
    """

    def __init__(self, num_classes, layer="preds"):
        self.num_classes = num_classes
        self.layer = layer
        self.confmat = ConfusionMatrix(num_classes=self.num_classes, normalize="true")
        super().__init__()

    def compute_error(self, src_val):
        src_label_preds = torch.softmax(src_val[self.layer], dim=-1).argmax(dim=-1)
        return (src_label_preds != src_val["labels"]).to(torch.float32).mean()

    def compute_distance(self, src_val, target_train):
        self.confmat = self.confmat.to(src_val[self.layer].device)
        src_confmat = self.confmat(src_val[self.layer], src_val["labels"]).to("cpu")

        tgt_label_preds = target_train[self.layer].argmax(dim=-1).to(torch.float32).to("cpu")
        tgt_histogram = torch.histogram(tgt_label_preds, bins=self.num_classes, density=False).hist / tgt_label_preds.size(0)

        denom = src_val["labels"].size(0)
        W = torch.tensor([src_val["labels"].eq(src_label).sum() / denom for src_label in tqdm(range(self.num_classes))])
        return (tgt_histogram - src_confmat).abs().sum(dim=1) @ W

    def compute_score(self, src_val, target_train):
        error = self.compute_error(src_val)
        distance = self.compute_distance(src_val, target_train)
        return 1. - ((error + distance).item() / 2.)


class HenryV3RegressionValidator(BaseValidator):
    """
    The sum of [ErrorValidator][pytorch_adapt.validators.ErrorValidator]
    and [L1Validator][pytorch_adapt.validators.L1Validator]
    src_val["labels"], src_val["preds"], target_train["preds"]
    """

    def __init__(self, layer="logits"):
        self.layer = layer
        super().__init__()

    def compute_distance(self, src_val, target_train):
        kde = KDEMultivariate(
            data=target_train[self.layer].cpu().numpy(), var_type=["c"] * target_train[self.layer].shape[-1], bw="normal_reference"
        )
        eps = torch.pdist(src_val[self.layer], p=torch.inf).min().item() / 2
        print("epsilon:", eps)

        y_hat = src_val[self.layer].cpu().numpy()
        s = (2 - 2 * kde.cdf(y_hat + eps) + 2 * kde.cdf(y_hat - eps)).sum() / y_hat.shape[0]
        return s

    def compute_error(self, src_val):
        return mse_loss(src_val[self.layer], src_val["labels"], reduction="mean")

    def compute_score(self, src_val, target_train):
        error = self.compute_error(src_val)
        distance = self.compute_distance(src_val, target_train)
        return 1. - (error + distance).item()


class HenryUpdatedValidator(BaseValidator):
    """
    The sum of [ErrorValidator][pytorch_adapt.validators.ErrorValidator]
    and [L1Validator][pytorch_adapt.validators.L1Validator]
    src_val["labels"], src_val["preds"], target_train["preds"]
    """

    def __init__(self, num_classes, src_val_at_init, layer="preds"):
        self.num_classes = num_classes
        self.src_val_at_init = src_val_at_init
        self.layer = layer
        self.confmat = ConfusionMatrix(num_classes=self.num_classes, normalize="true")
        super().__init__()

    def compute_error(self, src_val):
        src_label_preds = torch.softmax(src_val[self.layer], dim=-1).argmax(dim=-1)
        return (src_label_preds != src_val["labels"]).to(torch.float32).mean()

    def compute_domain_distance(self, src_val, target_train):
        self.confmat = self.confmat.to(src_val[self.layer].device)
        src_confmat = self.confmat(src_val[self.layer], src_val["labels"]).to("cpu")

        tgt_label_preds = torch.softmax(target_train[self.layer], dim=-1).argmax(dim=-1).to(torch.float32).to("cpu")
        tgt_histogram = torch.histogram(tgt_label_preds, bins=self.num_classes, density=False).hist / tgt_label_preds.size(0)

        return (src_confmat.T - tgt_histogram).abs().max()

    def compute_model_distance(self, src_val):
        self.confmat = self.confmat.to(src_val[self.layer].device)

        current_confmat = self.confmat(src_val[self.layer], src_val["labels"]).to("cpu")
        at_init_confmat = self.confmat(self.src_val_at_init[self.layer].to(src_val[self.layer].device), self.src_val_at_init["labels"]).to("cpu")

        d = []
        for pred_label in range(self.num_classes):
            for src_label in range(self.num_classes):
                _d = (at_init_confmat[src_label, pred_label] - current_confmat[src_label, pred_label]).abs()
                d.append(_d)
        return torch.stack(d).max()

    def compute_score(self, src_val, target_train):
        error = self.compute_error(src_val)
        domain_distance = self.compute_domain_distance(src_val, target_train)
        model_distance = self.compute_model_distance(src_val)
        return 1. - ((error + domain_distance + model_distance).item() / 3.)



if __name__ == "__main__":
    src_val = {
        "labels": torch.rand((5000, 4)),
        "logits": torch.rand((5000, 4)),
    }
    target_train = {
        "logits": torch.rand((5000, 4)),
    }
    v = HenryV3RegressionValidator()
    henry_score = v(src_val=src_val, target_train=target_train)
    # error term becomes: 0.25
    # distance term becomes: 0.5
    print(f"henry_score: {henry_score}")
