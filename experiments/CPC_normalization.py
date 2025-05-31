import sys
import os

# Add the project root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import subprocess
import random
import string

if __name__ == '__main__':
    # for seed in set(range(20)) - set([3,4]):
    for seed in range(5):
        print(f'seed: {seed}')
        # run system command with python subprocess
        # os.system(f'python main.py --seed {seed} --config runs/config_CPC.yaml')
        # generate a random 8 letter id string
        task = 'CPC'
        recording_methods = ['all', 'channel_wise', 'None'] 
        window_methods = ['all', 'channel_wise', 'None']
        for recording_norm in recording_methods:
            for window_norm in window_methods:
                wandb_id = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
                print(f'wandb_id: {wandb_id}')
                experiment_name = task
                experiment_name += f'-recording_norm_{recording_norm}'
                experiment_name += f'-window_norm_{window_norm}'
                experiment_name += f'-seed_{seed}'
                
                if recording_norm == 'None':
                    cache_dir = 'data'
                elif recording_norm == 'all':
                    cache_dir = 'data-no_cz-robust_scaled'
                elif recording_norm == 'channel_wise':
                    cache_dir = 'data-hp_0.1-robust_recording_channelwise'
                

                command_args = ['--config', f'runs/config_{task}.yaml', 
                    '--seed_everything', str(seed), 
                    # '--trainer.logger', 'null',
                    # '--trainer.max_epochs', '1',
                    '--trainer.logger.init_args.project', 'normalization',
                    '--trainer.logger.init_args.name', experiment_name,
                    '--trainer.logger.init_args.id', wandb_id,
                    '--data.val_release', 'ds005510',
                    '--data.cache_dir', cache_dir,
                    '--model.seed', str(seed),
                    '--model.init_args.window_norm', window_norm]

                command = ['python3', 'main.py']
                subprocess.run(command + ['validate'] + command_args)
                subprocess.run(command + ['fit'] + command_args)