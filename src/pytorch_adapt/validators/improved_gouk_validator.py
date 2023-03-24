import torch
from torchmetrics import ConfusionMatrix

from .base_validator import BaseValidator


class ImprovedGoukValidator(BaseValidator):
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

    def compute_distance(self, src_val):
        self.confmat = self.confmat.to(src_val[self.layer].device)
        src_confmat = self.confmat(src_val[self.layer], src_val["labels"]).to("cpu")

        d = []
        denom = src_val["labels"].size(0)
        for true_label in range(self.num_classes): # loop over true label values, y
            for other_label in range(self.num_classes): # loop again over true label values, y
                l1_norm = torch.norm(src_confmat[true_label] - src_confmat.mean(dim=0), p=1)
            d.append(l1_norm * src_val["labels"].argmax(dim=-1).eq(true_label).sum() / denom)
        return torch.stack(d).sum()

    def compute_score(self, src_val):
        error = self.compute_error(src_val)
        distance = self.compute_distance(src_val)
        return 1. - ((error + distance).item() / 2.)


class GoukUpdatedValidator(BaseValidator):
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

    def compute_domain_distance(self, src_val):
        self.confmat = self.confmat.to(src_val[self.layer].device)
        src_confmat = self.confmat(src_val[self.layer], src_val["labels"]).to("cpu")

        d = []
        for true_label in range(self.num_classes): # loop over true label values, y
            for other_label in list(set(range(self.num_classes)) - {true_label}): # loop over other classes, apart from y
                _d = torch.norm(src_confmat[true_label] - src_confmat[other_label], p=torch.inf)
                d.append(_d)
        return torch.stack(d).min() # take minimum over whole set

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

    def compute_score(self, src_val):
        error = self.compute_error(src_val)
        domain_distance = self.compute_domain_distance(src_val)
        model_distance = self.compute_model_distance(src_val)
        return 1. - ((error + domain_distance + model_distance).item() / 3.)



if __name__ == "__main__":
    from pytorch_adapt.validators import GoukValidator
    src_val = {
        "labels": torch.tensor([0, 1, 1, 0]),
        "preds": torch.tensor([
            [2.0, -2.],
            [0.5, 1.3],
            [0.0, 9.0],
            [0.1, 0.5]
        ])
        # predicted classes will be: [0, 1, 1, 1]
    }
    v = GoukValidator(layer="preds", num_classes=2)
    gouk_score = v(src_val=src_val)
    # error term becomes: 0.25
    # distance term becomes: 0.5
    print(f"gouk_score: {gouk_score}")
