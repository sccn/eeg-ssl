import sys
# sys.path.insert(0, "../")
# import matplotlib.pyplot as plt
# import numpy as np
import os

from libs import ssl_dataloader as dataloader
from libs import ssl_model as ssl_model

import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim




USE_GPU = True

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('using device:', device)

import time

rand_seed = 50
torch.manual_seed(rand_seed)

base_dir = "/expanse/projects/nemar/dtyoung/eeg-ssl"
ssl_task = "RP"
bs = 32
lr = 0.003
log_dir = f"{ssl_task}/seed-{rand_seed}/batch-size-{bs}_lr-{lr}"
writer = SummaryWriter(f"{base_dir}/logs/runs/{log_dir}")
model_save_dir = f"{base_dir}/logs/checkpoints/{log_dir}"
if not os.path.isdir(model_save_dir):
    os.mkdir(model_save_dir)
    
start = time.time()
dataset = dataloader.BIDSDataset(
    x_params={
        "feature": "SSLTransform", 
        "cache_dir": "/expanse/projects/nemar/dtyoung/deep-eeg/cache",
        "method": ssl_task,
        "win": 256,
        "stride": 128,
        "tau_pos": 256*3,
        "tau_neg": 256*3,
        "n_samples": 1,
        "seed": rand_seed,        
    },
    seed=rand_seed,
    n_sessions=2
)
print("Elapsed time:", time.time()-start)
# from matplotlib import pyplot as plt
# plt.imshow(dataset[10][0][0][0,:,:].cpu())

model = ssl_model.VGGSSL(ssl_task).to(device=device)
training_data, validation_data, test_data = random_split(dataset, [0.7, 0.2, 0.1], generator=torch.Generator().manual_seed(42))
loader_train = DataLoader(training_data, batch_size = bs, shuffle = True)
loader_val = DataLoader(validation_data, batch_size = bs, shuffle = True)
optimizer = optim.Adam(model.parameters(), lr = lr)

num_epochs = 100
print_every = 10

model.train(num_epochs, print_every, loader_train, model_save_dir, optimizer, writer)