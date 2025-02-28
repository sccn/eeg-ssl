import subprocess
import numpy as np

windows_lens = np.arange(0.5,10.5,0.5)

for window_len in windows_lens:
    subprocess.run(["python3", "main.py", "fit", "--config", "runs/config_RP.yaml", "--data.ssl_task.n_samples_per_dataset", "1", "--trainer.logger", "null", "--trainer.max_epochs", "1", "--data.window_len_s", str(window_len)])

    
