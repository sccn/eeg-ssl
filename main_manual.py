from libs.ssl_data import SSLHBNDataModule
from libs.ssl_utils import LitSSL
from libs.ssl_task import RelativePositioning, Classification
from libs.ssl_model import VGGSSL
from lightning.pytorch.cli import LightningCLI
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from braindecode.models import ShallowFBCSPNet

from braindecode.datautil import load_concat_dataset
from braindecode.preprocessing.windowers import create_fixed_length_windows
from braindecode.datasets.base import BaseConcatDataset
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torchmetrics.functional.classification import binary_accuracy, accuracy
import yaml

if __name__ == '__main__':
    # load config from runs/config_CPC.yaml
    with open('../runs/config_CPC.yaml', 'r') as f:
        config = yaml.safe_load(f)