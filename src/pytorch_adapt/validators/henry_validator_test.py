import torch
from torchmetrics import ConfusionMatrix

from pytorch_adapt.validators.base_validator import BaseValidator


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
        print(src_val[self.layer], src_val["labels"])
        self.confmat = self.confmat.to(src_val[self.layer].device)
        src_confmat = self.confmat(src_val[self.layer], src_val["labels"]).to("cpu")
        print(f"confmat: {src_confmat}")

        tgt_label_preds = torch.softmax(target_train[self.layer], dim=-1).argmax(dim=-1).to(torch.float32).to("cpu")
        tgt_histogram = torch.histogram(tgt_label_preds, bins=self.num_classes, density=False).hist / tgt_label_preds.size(0)

        d = []
        for pred_label in range(self.num_classes):
            for src_label in range(self.num_classes):
                _d = (tgt_histogram[pred_label] - src_confmat[src_label, pred_label]).abs()
                d.append(_d)
        return torch.stack(d).max()

    def compute_score(self, src_val, target_train):
        error = self.compute_error(src_val)
        distance = self.compute_distance(src_val, target_train)
        print(error.item(), distance.item(), (error + distance).item())
        return 1. - ((error + distance).item() / 2.)



if __name__ == "__main__":
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
    target_train = {
        "preds": torch.tensor([
            [2.0, -2.],
            [1.3, 0.5],
            [0.0, 9.0],
            [0.1, 0.5]
        ])
        # predicted classes will be: [0, 0, 1, 1]
    }
    v = HenryValidator(layer="preds", num_classes=2)
    henry_score = v(src_val=src_val, target_train=target_train)
    # error term becomes: 0.25
    # distance term becomes: 0.5
