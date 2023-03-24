from .classifier import Classifier, LinearClassifier
from .regressor import Regressor
from .segmenter import Segmenter
from .discriminator import Discriminator
from .mnist import MNISTFeatures
from .unet import UNetG, UNetC
from .deeplab import DeepLabV3G, DeepLabV3C, DeepLabV3D
from .pretrained import (
    domainnet126C,
    domainnet126G,
    mnistC,
    mnistG,
    office31C,
    office31G,
    officehomeC,
    officehomeG,
    voc_multilabelC,
    voc_multilabelG,
    resnet50_timm,
    resnet18,
    resnet50,
    resnet101
)
from .pretrained_scores import pretrained_src_accuracy, pretrained_target_accuracy
