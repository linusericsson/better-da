from torchmetrics import JaccardIndex

from .torchmetrics_validator import TorchmetricsValidator


class IOUValidator(TorchmetricsValidator):
    """
    Returns IOU using the
    [torchmetrics jaccard_index function](https://torchmetrics.readthedocs.io/en/stable/classification/jaccard_index.html#functional-interface).

    The required dataset splits are ```["src_val"]```.
    This can be changed using [```key_map```][pytorch_adapt.validators.BaseValidator.__init__].
    """

    def __init__(self, num_classes):
        super().__init__()
        self.jaccard_index = JaccardIndex(task="multiclass", num_classes=num_classes)

    @property
    def accuracy_fn(self):
        return self.jaccard_index
