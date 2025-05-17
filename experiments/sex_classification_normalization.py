import sys
import os

# Add the project root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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
        channel_wise_norms = [True, False]
        global_norms = [True, False]
        for global_norm in global_norms:
            for channel_wise_norm in channel_wise_norms:
                wandb_id = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
                print(f'wandb_id: {wandb_id}')

                subprocess.run(['python3', 'main.py', 'validate', '--config', f'runs/config_{task}.yaml', 
                                '--seed_everything', str(seed), 
                                '--model.seed', str(seed),
                                '--trainer.logger.init_args.id', wandb_id,
                                '--data.cache_dir', 'data-no_cz-robust_scaled' if global_norm else 'data',
                                '--model.init_args.channel_wise_norm', str(channel_wise_norm),
                                '--model.init_args.encoder_kwargs.n_chans', '128' if global_norm else '129',])

                subprocess.run(['python3', 'main.py', 'fit', '--config', f'runs/config_{task}.yaml', 
                                '--seed_everything', str(seed), 
                                '--model.seed', str(seed),
                                '--trainer.max_epochs', '6',
                                '--trainer.logger.init_args.id', wandb_id,
                                '--data.cache_dir', 'data-no_cz-robust_scaled' if global_norm else 'data',
                                '--model.init_args.channel_wise_norm', str(channel_wise_norm),
                                '--model.init_args.encoder_kwargs.n_chans', '128' if global_norm else '129',])