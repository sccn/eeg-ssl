import sys
sys.path.insert(0,'../')
from libs.ssl_dataloader import *
from libs.ssl_model import *
from libs.ssl_utils import *
from libs.eeg_utils import *
from braindecode.preprocessing import (
    preprocess, Preprocessor, create_fixed_length_windows)
from braindecode.datasets import BaseDataset, BaseConcatDataset, WindowsDataset
from braindecode.preprocessing.windowers import EEGWindowsDataset
from braindecode.datautil import load_concat_dataset
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

datasets = []
releases = list(range(9,0,-1))
hbn_datasets = ['ds005514','ds005512','ds005511','ds005510','ds005509','ds005508','ds005507','ds005506','ds005505']
hbn_release_ds = dict(zip(releases,hbn_datasets))

ds1 = HBNDataset(hbn_release_ds[1], tasks=['RestingState'], num_workers=-1)
ds2 = HBNDataset(hbn_release_ds[6], tasks=['RestingState'], num_workers=-1)

all_ds = BaseConcatDataset([ds1, ds2])

all_ds.save('data/hbn', overwrite=True)