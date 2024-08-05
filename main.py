import os
import sys
import wandb
sys.path.insert(0,'../')
from libs.ssl_dataloader import *
from libs.ssl_model import *
from libs.ssl_utils import *
from libs import eeg_utils
from sklearn.model_selection import train_test_split
import argparse

def run_experiment(args):
    if not args.debug:
        wandb.init(
            # Set the project where this run will be logged
            project="hbn-ssl", 
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            #   name=f"experiment_{run}", 
            # Track hyperparameters and run metadata
            config={
                "seed": args.seed,
                "nsubjects": args.nsubjects,
                "window": args.window,
                'mask_prob': args.mask_prob,
                'batch_size': args.batch_size,
                'epochs': args.epochs
            })
    SFREQ = 128
    dataset = MaskedContrastiveLearningDataset(
        data_dir = args.data,
        # subjects = subj_train.tolist(),
        n_subjects=args.nsubjects,
        x_params = {
            'sfreq': SFREQ,
            'window': args.window
        },
        random_seed = args.seed
    )
    print('Length of dataset', len(dataset))
    print('X dim', dataset[0][0].shape)
    print('Y', dataset[0][1])
    # eeg_utils.plot_raw_eeg(dataset[5][0]) # ERROR

    model = Wav2VecBrainModel()
    task = MaskedContrastiveLearningTask(dataset, 
            task_params={
                'mask_prob': args.mask_prob
            },
            train_params={
                'num_epochs': args.epochs,
                'batch_size': args.batch_size,
                'print_every': args.print_every
            },
            random_seed=args.seed,
            debug=args.debug
    )
    trained_model = task.train(model)


def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="A simple command line argument parser")

    # Add arguments
    parser.add_argument('--data', type=str, default="/mnt/nemar/child-mind-rest", help="Path to data directory (Default: /mnt/nemar/child-mind-rest)")
    parser.add_argument('--verbose', action='store_true', help="Increase output verbosity")
    parser.add_argument('--seed', type=int, default=9, help="Random seed (default: 9)")
    parser.add_argument('--nsubjects', type=int, default=50, help="Number of subject recordings to be used for dataset (default: 50)")
    parser.add_argument('--window', type=int, default=5, help="EEG window size in second(s) (default: 50)")
    parser.add_argument('--mask_prob', type=float, default=0.3, help="Masking probability (default: 0.3)")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs (default: 10)")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size (default: 128)")
    parser.add_argument('--print_every', type=int, default=100, help="Display model performance every # training step (default: 100)")
    parser.add_argument('--debug', type=bool, default=True, help="Whether running in debug mode without wandb tracking")

    # Parse the arguments
    args = parser.parse_args()

    wandb.login()
    run_experiment(args)

if __name__ == "__main__":
    main()