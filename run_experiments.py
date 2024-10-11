import os
from dotenv import load_dotenv
import sys
import wandb
sys.path.insert(0,'../')
from libs.ssl_dataloader import *
from libs.ssl_model import *
from libs.ssl_utils import *

if __name__ == '__main__':
    load_dotenv()
    WANDB_API_KEY = os.environ['WANDB_API_KEY']
    x_params = {
            'sfreq': 128,
            'window': 5,
    }
    train_params={
            'num_epochs': 100,
            'batch_size': 128,
            'print_every': 1,
            'learning_rate': 0.001,
    }
    task_params={
            'task': 'RelativePositioning',
            'win': 50,
            'tau_pos': 100,
            'tau_neg': 100,
            'n_samples': 1,
    }
    # combine all the parameters into a single config dict for logging
    config = {**x_params, **train_params, **task_params}

    dataset = HBNRestBIDSDataset(
            data_dir = "/mnt/nemar/openneuro/ds004186",
            x_params = x_params,
    )
    config['dataset'] = 'ds004186'

    model = VGGSSL()
    config['model'] = 'VGGSSL'
    wandb.init(
        # Set the project where this run will be logged
        project="ssl-hbn-rest", 
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        # name=f"experiment_{run}", 
        # Track hyperparameters and run metadata
        config=config)

    trainer = Trainer(
        dataset=dataset,
        model=model,
        train_params=task_params,
        task_params=task_params,
        wandb=wandb
    )
    trainer.train()