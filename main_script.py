from libs.ssl_data import SSLHBNDataModule
from libs.ssl_utils import LitSSL
from libs.ssl_task import RelativePositioning, Classification, CPC
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
import subprocess
import random
import string

if __name__ == '__main__':
    # for seed in set(range(20)) - set([3,4]):
    for seed in [3]:
        print(f'seed: {seed}')
        # run system command with python subprocess
        # os.system(f'python main.py --seed {seed} --config runs/config_CPC.yaml')
        # generate a random 8 letter id string
        task = 'Classification'
        wandb_id = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
        print(f'wandb_id: {wandb_id}')
        subprocess.run(['python3', 'main.py', 'validate', '--config', f'runs/config_{task}.yaml', 
                        '--seed_everything', str(seed), 
                        '--model.seed', str(seed),
                        '--trainer.logger.init_args.id', wandb_id,])

        subprocess.run(['python3', 'main.py', 'fit', '--config', f'runs/config_{task}.yaml', 
                        '--seed_everything', str(seed), 
                        '--model.seed', str(seed),
                        '--trainer.logger.init_args.id', wandb_id,])