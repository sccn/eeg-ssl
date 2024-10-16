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

    seed = 0
    x_params = {
        'sfreq': 128,
        'window': 20,
        'preprocess': False,
    }
    train_params={
        'num_epochs': 10,
        'batch_size': 64,
        'print_every': 1,
        'learning_rate': 0.00001,
        'num_workers': 10,
    }
    task_params={
        'task': 'RelativePositioning',
        'sfreq': 128,
        'win': 0.5,
        'tau_pos': 10,
        'tau_neg': 10,
        'n_samples': 1,
        'seed': seed
    }
    # combine all the parameters into a single config dict for logging
    config = {**x_params, **train_params, **task_params}

    dataset = HBNRestBIDSDataset(
            data_dir = "/mnt/nemar/openneuro/ds004186",
            x_params = x_params,
            random_seed=seed,
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
        config=config,
        resume="allow")

    trainer = Trainer(
        dataset=dataset,
        model=model,
        train_params=train_params,
        task_params=task_params,
        wandb=wandb,
        seed=seed,
    )
    config['seed'] = seed

    trainer.train()