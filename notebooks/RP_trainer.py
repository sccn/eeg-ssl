import sys
sys.path.insert(0, '../')
from libs.ssl_dataloader import *
from libs.ssl_model import *
from libs.ssl_utils import *
from libs.eeg_utils import *
from libs.evaluation import train_regressor, RankMe
from braindecode.preprocessing import (
    preprocess, Preprocessor, create_fixed_length_windows)
from braindecode.datasets import BaseDataset, BaseConcatDataset, WindowsDataset
from braindecode.datautil import load_concat_dataset
import numpy as np
from sklearn.model_selection import train_test_split
from braindecode.datasets import BaseConcatDataset
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import lightning as L
import torch
from torch import nn
from braindecode.models import ShallowFBCSPNet
from braindecode.samplers import RelativePositioningSampler

def load_data(window_len_s, tau_pos_s, tau_neg_s=None, same_rec_neg=False, random_state=9):
    if not os.path.exists('data/hbn_preprocessed'):
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

        from sklearn.preprocessing import scale as standard_scale

        os.makedirs('data/hbn_preprocessed', exist_ok=True)

        sampling_rate = 250 # resample to follow the tutorial sampling rate
        high_cut_hz = 59
        # Factor to convert from V to uV
        factor = 1e6
        preprocessors = [
            Preprocessor(lambda data: np.multiply(data, factor)),  # Convert from V to uV
            Preprocessor('crop', tmin=10),  # crop first 10 seconds as begining of noise recording
            Preprocessor('filter', l_freq=None, h_freq=high_cut_hz),
            Preprocessor('resample', sfreq=sampling_rate),
            Preprocessor('notch_filter', freqs=(60, 120)),
            Preprocessor(standard_scale, channel_wise=True),
        ]

        # Transform the data
        preprocess(all_ds, preprocessors, save_dir='data/hbn_preprocessed', overwrite=True, n_jobs=-1)
    else:
        all_ds = load_concat_dataset(path='data/hbn_preprocessed', preload=False)

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


    subjects = np.unique(windows_ds.description['subject'])
    subj_train, subj_test = train_test_split(
        subjects, test_size=0.4, random_state=random_state)
    subj_valid, subj_test = train_test_split(
        subj_test, test_size=0.5, random_state=random_state)

    split_ids = {'train': subj_train, 'valid': subj_valid, 'test': subj_test}
    splitted = dict()
    for name, values in split_ids.items():
        splitted[name] = RelativePositioningDataset(
            [ds for ds in windows_ds.datasets
            if ds.description['subject'] in values])

    print('n train datasets:', len(splitted['train'].datasets))
    print('n validation datasets:', len(splitted['valid'].datasets))
    print('n test datasets:', len(splitted['test'].datasets))

    def get_min_sample_per_dataset(windows_ds: BaseConcatDataset):
        subjects = windows_ds.get_metadata()['subject'].values
        _, counts = np.unique(subjects, return_counts=True)
        return np.min(counts)

    n_samples_per_dataset = get_min_sample_per_dataset(windows_ds) # this number is a function of window_len_s and recording length
    sfreq = windows_ds.datasets[0].raw.info['sfreq']
    tau_pos, tau_neg = int(sfreq * tau_pos_s), int(sfreq * tau_neg_s) if tau_neg_s else int(sfreq * 2 * tau_pos_s)
    n_examples_train = n_samples_per_dataset * len(splitted['train'].datasets)
    n_examples_valid = n_samples_per_dataset * len(splitted['valid'].datasets)
    n_examples_test = n_samples_per_dataset * len(splitted['test'].datasets)
    print('n train samples:', n_examples_train)
    print('n validation samples:', n_examples_valid)
    print('n test samples:', n_examples_test)

    train_sampler = RelativePositioningSampler(
        splitted['train'].get_metadata(), tau_pos=tau_pos, tau_neg=tau_neg,
        n_examples=n_examples_train, same_rec_neg=same_rec_neg, random_state=random_state)
    valid_sampler = RelativePositioningSampler(
        splitted['valid'].get_metadata(), tau_pos=tau_pos, tau_neg=tau_neg,
        n_examples=n_examples_valid, same_rec_neg=same_rec_neg,
        random_state=random_state).presample()
    test_sampler = RelativePositioningSampler(
        splitted['test'].get_metadata(), tau_pos=tau_pos, tau_neg=tau_neg,
        n_examples=n_examples_test, same_rec_neg=same_rec_neg,
        random_state=random_state).presample()
    
    return (splitted['train'], train_sampler), (splitted['valid'], valid_sampler), (splitted['test'], test_sampler)

    
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
        self.rankme = RankMe()

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
        X, Y, _ = batch
        z = self.embed(X)
        self.rankme.update(z)

        z = z.float().detach().cpu().numpy()
        Y = Y.float().detach().cpu().numpy()
        from sklearn import linear_model, neural_network
        regr = linear_model.LinearRegression()
        regr, linear_score = train_regressor(regr, z, Y)
        self.log('val_linear_score', linear_score)
        regr = neural_network.MLPRegressor(max_iter=1000)
        regr, nn_score = train_regressor(regr, z, Y)
        self.log('val_nn_score', nn_score)
        
    def test_step(self, batch, batch_idx):
        # this is the test loop
        X, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        test_loss = F.mse_loss(x_hat, x)
        self.log("test_loss", test_loss)

    def on_validation_epoch_end(self):
        # log epoch metric
        self.log('val_rankme', self.rankme.compute())

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()

    # Trainer arguments
    parser.add_argument("--window_len_s", type=int, default=10)
    parser.add_argument("--tau_pos_s", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", type=str, default='gpu')
    parser.add_argument("--num_workers", type=int, default=4)

    # Parse the user inputs and defaults (returns a argparse.Namespace)
    args = parser.parse_args()

    (train_ds, train_sampler), (valid_ds, valid_sampler), (test_ds, test_sampler) = load_data(args.window_len_s, args.tau_pos_s)
    train_loader = DataLoader(train_ds, sampler=train_sampler, batch_size=args.batch_size, num_workers=args.num_workers)
    valid_ds.return_pair = False
    val_loader = DataLoader(valid_ds, batch_size=args.batch_size, num_workers=args.num_workers)

    # Extract number of channels and time steps from dataset
    i = next(iter(train_loader)) # ((BxCxT, pair2), labels)
    n_channels, input_size_samples = i[0][0][0].shape
    emb_size = 100
    model = LitSSL(n_channels, train_ds.datasets[0].raw.info['sfreq'], input_size_samples, args.window_len_s, emb_size)

    # Use the parsed arguments in your program
    if args.device == 'hpu':
        from lightning_habana.pytorch.accelerator import HPUAccelerator
        accelerator = HPUAccelerator()
        trainer = L.Trainer(max_epochs=args.epochs, accelerator=accelerator, devices='auto', strategy='auto')
    else:
        trainer = L.Trainer(max_epochs=args.epochs, accelerator=args.device)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader) #, ckpt_path="lightning_logs/version_10/checkpoints/epoch=199-step=20000.ckpt")
