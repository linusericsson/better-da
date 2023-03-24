from .base_dataset import BaseDataset, BaseDownloadableDataset
from .clipart1k import Clipart1kMultiLabel
from .combined_source_and_target import CombinedSourceAndTargetDataset
from .concat_dataset import ConcatDataset
from .dataloader_creator import DataloaderCreator
from .domainnet import DomainNet, DomainNet126, DomainNet126Full
from .getters import (
    get_domainnet126,
    get_mnist_mnistm,
    get_mnist,
    get_mnistmr,
    get_mnistms,
    get_office31,
    get_officehome,
    get_voc_multilabel,
    get_visda2017c,
    get_biwi,
    get_faces300w,
    get_dogs_and_birds,
    get_animal_pose,
    get_synthia_cityscapes,
    get_gta5_cityscapes,
)
from .mnistm import MNISTM
from .office31 import Office31, Office31Full
from .biwi import BIWI, BIWIFull
from .faces300w import Faces300W, Faces300WFull
from .officehome import OfficeHome, OfficeHomeFull
from .urban_scenes import SYNTHIACityscapes, GTA5Cityscapes
from .gta5 import GTA5
from .synthia import SYNTHIA
from .cityscapes import Cityscapes
from .pseudo_labeled_dataset import PseudoLabeledDataset
from .source_dataset import SourceDataset
from .target_dataset import TargetDataset
from .voc_multilabel import VOCMultiLabel
from .visda2017_classification import VisDA2017Classification
