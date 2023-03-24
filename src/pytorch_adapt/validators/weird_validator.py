import torch
from torchmetrics import ConfusionMatrix

from .base_validator import BaseValidator


class WeirdValidator(BaseValidator):
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
        for pred_label in range(self.num_classes): # loop over predicton values, y_hat
            for true_label in range(self.num_classes): # loop over true label values, y
                for i in list(set(range(self.num_classes)) - {true_label}): # loop over other classes, apart from y
                    _d = (src_confmat[true_label, pred_label] - src_confmat[i, pred_label]).abs()
                    d.append(_d)
        return torch.stack(d).min() # take minimum over whole set

    def compute_score(self, src_val):
        error = self.compute_error(src_val)
        distance = self.compute_distance(src_val)
        return 1. - ((error + distance).item() / 2.)



if __name__ == "__main__":
    from pytorch_adapt.validators import WeirdValidator
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
    v = WeirdValidator(layer="preds", num_classes=2)
    gouk_score = v(src_val=src_val)
    # error term becomes: 0.25
    # distance term becomes: 0.5
    print(f"gouk_score: {gouk_score}")
