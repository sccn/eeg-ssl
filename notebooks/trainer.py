import os
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

random_state = 87
window_len_s = 10

if not os.path.exists('data/hbn_preprocessed_windowed_scaled'):
    datasets = []
    releases = list(range(9,0,-1))
    hbn_datasets = ['ds005514','ds005512','ds005511','ds005510','ds005509','ds005508','ds005507','ds005506','ds005505']
    hbn_release_ds = dict(zip(releases,hbn_datasets))

    if not os.path.exists('data'):
        os.makedirs('data', exist_ok=True)
    if not os.path.exists('data/ds005510'):
        # download zip file from google drive and put it in data folder
        # https://drive.google.com/file/d/1KWEDoZOqyLojq0hQx8lUNTWSdZ5tBlTc/view?usp=sharing
        import zipfile
        with zipfile.ZipFile('data/ds005510.zip', 'r') as zip_ref:
            zip_ref.extractall('data')
    # make sure you downloaded ds005505 and placed it in data folder
    ds2 = HBNDataset(hbn_release_ds[6], tasks=['RestingState'], num_workers=-1, preload=False, data_path='data')

    all_ds = BaseConcatDataset([ds2]) # [ds1, ds2]

    from numpy import multiply
    from sklearn.preprocessing import scale as standard_scale

    os.makedirs('data/hbn_preprocessed', exist_ok=True)

    sampling_rate = 250 # resample to follow the tutorial sampling rate
    high_cut_hz = 59
    # Factor to convert from V to uV
    factor = 1e6
    preprocessors = [
        #Preprocessor(lambda data: multiply(data, factor)),  # Convert from V to uV
        Preprocessor('crop', tmin=10),  # crop first 10 seconds as begining of noise recording
        Preprocessor('filter', l_freq=None, h_freq=high_cut_hz),
        Preprocessor('resample', sfreq=sampling_rate),
        Preprocessor('notch_filter', freqs=(60, 120)),
        Preprocessor(standard_scale, channel_wise=True),
    ]

    # Transform the data
    preprocess(all_ds, preprocessors, save_dir='data/hbn_preprocessed', overwrite=True, n_jobs=-1)

    target_name = 'age'
    for ds in all_ds.datasets:
        ds.target_name = target_name

    fs = all_ds.datasets[0].raw.info['sfreq']
    print('sampling rate', fs)
    window_len_samples = int(fs * window_len_s)
    window_stride_samples = int(fs * window_len_s) # non-overlapping
    windows_ds = create_fixed_length_windows(
        all_ds, start_offset_samples=0, stop_offset_samples=None,
        window_size_samples=window_len_samples,
        window_stride_samples=window_stride_samples, drop_last_window=True,
        preload=False)

    os.makedirs('data/hbn_preprocessed_windowed_scaled', exist_ok=True)
    windows_ds.save('data/hbn_preprocessed_windowed_scaled', overwrite=True)
else:
    windows_ds = load_concat_dataset(path='data/hbn_preprocessed_windowed_scaled', preload=False)

import numpy as np
from sklearn.model_selection import train_test_split
from braindecode.datasets import BaseConcatDataset

subjects = np.unique(windows_ds.description['subject'])
subj_train, subj_test = train_test_split(
    subjects, test_size=0.4, random_state=random_state)
subj_valid, subj_test = train_test_split(
    subj_test, test_size=0.5, random_state=random_state)

class RelativePositioningDataset(BaseConcatDataset):
    """BaseConcatDataset with __getitem__ that expects 2 indices and a target.
    """

    def __init__(self, list_of_ds):
        super().__init__(list_of_ds)
        self.return_pair = True

    def __getitem__(self, index):
        if self.return_pair:
            ind1, ind2, y = index
            return (super().__getitem__(ind1)[0],
                    super().__getitem__(ind2)[0]), y
        else:
            return super().__getitem__(index)

    @property
    def return_pair(self):
        return self._return_pair

    @return_pair.setter
    def return_pair(self, value):
        self._return_pair = value

split_ids = {'train': subj_train, 'valid': subj_valid, 'test': subj_test}
splitted = dict()
for name, values in split_ids.items():
    splitted[name] = RelativePositioningDataset(
        [ds for ds in windows_ds.datasets
         if ds.description['subject'] in values])

print('train datasets', len(splitted['train'].datasets))
print('validation datasets', len(splitted['valid'].datasets))
print('test datasets', len(splitted['test'].datasets))

from braindecode.samplers import RelativePositioningSampler

sfreq = 250
tau_pos, tau_neg = int(sfreq * 10), int(sfreq * 2 * 10)
n_examples_train = 250 * len(splitted['train'].datasets)
n_examples_valid = 250 * len(splitted['valid'].datasets)
n_examples_test = 250 * len(splitted['test'].datasets)

train_sampler = RelativePositioningSampler(
    splitted['train'].get_metadata(), tau_pos=tau_pos, tau_neg=tau_neg,
    n_examples=n_examples_train, same_rec_neg=False, random_state=random_state)
valid_sampler = RelativePositioningSampler(
    splitted['valid'].get_metadata(), tau_pos=tau_pos, tau_neg=tau_neg,
    n_examples=n_examples_valid, same_rec_neg=False,
    random_state=random_state).presample()
test_sampler = RelativePositioningSampler(
    splitted['test'].get_metadata(), tau_pos=tau_pos, tau_neg=tau_neg,
    n_examples=n_examples_test, same_rec_neg=False,
    random_state=random_state).presample()

import lightning as L
import torch
from torch import nn
from braindecode.util import set_random_seeds
from braindecode.models import ShallowFBCSPNet
# define the LightningModule
class LitSSL(L.LightningModule):
    def __init__(self, n_channels, sfreq, input_size_samples, window_len_s, emb_size, dropout=0.5):
        super().__init__()
        self.emb = VGGSSL() # self.create_embedding_layer(n_channels, sfreq, input_size_samples, window_len_s)
        self.pooling = nn.AdaptiveAvgPool2d(32)
        self.clf = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(1024, emb_size),
            nn.Dropout(dropout),
            nn.Linear(emb_size, 1)
        )

    def create_embedding_layer(self, n_channels, sfreq, input_size_samples, window_len_s):
        return ShallowFBCSPNet(
            n_chans=n_channels,
            sfreq=sfreq,
            n_outputs=emb_size,
            # n_conv_chs=16,
            n_times=input_size_samples,
            input_window_seconds=window_len_s,
            # dropout=0,
            # apply_batch_norm=True,
        )

    def embed(self, x):
        z = self.clf[1](self.pooling(self.emb(x)).flatten(start_dim=1))
        return z

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        X, y = batch
        x1, x2 = X[0], X[1]
        z1, z2 = self.emb(x1), self.emb(x2)
        z = self.pooling(torch.abs(z1 - z2)).flatten(start_dim=1)

        loss = nn.functional.binary_cross_entropy_with_logits(self.clf(z).flatten(), y)

        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        from sklearn import linear_model
        regr = linear_model.LinearRegression()
        X, Y, _ = batch
        z = self.embed(X).detach().cpu().numpy()
        Y = Y.detach().cpu().numpy()
        isnan = np.isnan(Y)
        embs = z[~isnan]
        labels = Y[~isnan]
        regr.fit(embs, labels)
        score = regr.score(embs, labels) 
        self.log('val_score', score)
        

    def test_step(self, batch, batch_idx):
        # this is the test loop
        X, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        test_loss = F.mse_loss(x_hat, x)
        self.log("test_loss", test_loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

# Extract number of channels and time steps from dataset
n_channels, input_size_samples = windows_ds[0][0].shape
emb_size = 100
classes = list(range(5))


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()

    # Trainer arguments
    parser.add_argument("--batch_size", type=int, default=128)

    # Parse the user inputs and defaults (returns a argparse.Namespace)
    args = parser.parse_args()

    model = LitSSL(n_channels, sfreq, input_size_samples, window_len_s, emb_size)

    train_loader = DataLoader(splitted['train'], sampler=train_sampler, batch_size=args.batch_size)
    splitted['valid'].return_pair = False
    val_loader = DataLoader(splitted['valid'], batch_size=args.batch_size)

    # Use the parsed arguments in your program
    trainer = L.Trainer(max_epochs=1, accelerator='cpu')
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader) #, ckpt_path="lightning_logs/version_10/checkpoints/epoch=199-step=20000.ckpt")