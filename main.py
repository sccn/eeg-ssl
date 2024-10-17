import os
from dotenv import load_dotenv
import sys
import wandb
sys.path.insert(0,'../')
from libs.ssl_dataloader import *
from libs.ssl_model import *
from libs.ssl_utils import *
import argparse

def run_experiment(args):
    load_dotenv()
    WANDB_API_KEY = os.environ['WANDB_API_KEY']

    seed = args.seed
    x_params = {
        'sfreq': 128,
        'window': args.sample_window,
        'preprocess': args.preprocess,
    }
    train_params={
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'print_every': args.print_every,
        'learning_rate': args.lr,
        'num_workers': args.num_workers,
    }
    task_params={
        'task': args.task,
        'sfreq': 128,
        'win': args.window,
        'tau_pos': args.tau_pos,
        'tau_neg': args.tau_neg,
        'n_samples': args.n_samples,
        'seed': seed
    }
    # combine all the parameters into a single config dict for logging
    config = {**x_params, **train_params, **task_params}

    dataset = HBNRestBIDSDataset(
        data_dir = args.data,
        x_params = x_params,
        random_seed=seed,
    )
    config['dataset'] = args.dataset

    # instantiate the model using args.model string
    model = globals()[args.model]()
    config['model'] = args.model
    wandb.init(
        # Set the project where this run will be logged
        project="ssl-hbn-rest", 
        # id="relative-positioning-with-multiprocess-dataloader",
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        # name=f"experiment_{run}", 
        # Track hyperparameters and run metadata
        config=config,
        # resume="allow",
    )

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
    # trainer.train(checkpoint='/home/dung/eeg-ssl/wandb/run-20241016_111351-relative-positioning-with-multiprocess-dataloader/files/checkpoint_epoch-9')


def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="A simple command line argument parser")

    # Add arguments
    parser.add_argument('--data', type=str, default="/mnt/nemar/openneuro/ds004186", help="Path to data directory (Default: /mnt/nemar/openneuro/ds004186)")
    parser.add_argument('--dataset', type=str, default="ds004186", help="Dataset name (Default: ds004186)")
    parser.add_argument('--model', type=str, default="VGGSSL", help="Model name (Default: VGGSSL)")
    parser.add_argument('--sample_window', type=int, default=20, help="EEG window length in second(s) (default: 20)")
    parser.add_argument('--task', type=str, default="RelativePositioning", help="SSL task (Default: RelativePositioning)")
    parser.add_argument('--tau_pos', type=int, default=10, help="Positive window size in second(s) (default: 10)")
    parser.add_argument('--tau_neg', type=int, default=10, help="Negative window size in second(s) (default: 10)")
    parser.add_argument('--n_samples', type=int, default=1, help="Number of sample per recording (default: 1)")
    parser.add_argument('--preprocess', action='store_true', help="Whether to preprocess the data (Default: False)")
    parser.add_argument('--num_workers', type=int, default=0, help="Number of dataloader workers (default: 0)")
    parser.add_argument('--seed', type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument('--window', type=float, default=5, help="Task EEG segment length in second(s) (default: 5)")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs (default: 10)")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument('--lr', type=float, default=0.001, help="Adam learning rate")
    parser.add_argument('--print_every', type=int, default=1, help="Display model performance every # training step (default: 1)")
    parser.add_argument('--verbose', action='store_true', help="Increase output verbosity")
    parser.add_argument('--debug', action='store_true', help="Whether running in debug mode without wandb tracking")

    # Parse the arguments
    args = parser.parse_args()
    print('Arguments:', args)

    run_experiment(args)

if __name__ == "__main__":
    main()