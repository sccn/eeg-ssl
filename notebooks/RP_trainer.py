import sys
sys.path.insert(0, '../')
from libs.ssl_dataloader import *
from libs.ssl_model import *
from libs.ssl_utils import *
from libs.ssl_utils import DistributedRelativePositioningSampler
from libs.eeg_utils import *
from libs.evaluation import train_regressor, RankMe
from braindecode.preprocessing import (
    preprocess, Preprocessor, create_fixed_length_windows)
from braindecode.datasets import BaseConcatDataset
from braindecode.datautil import load_concat_dataset
from braindecode.models import ShallowFBCSPNet
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.cli import LightningCLI

class RelativePositioningHBNDataModule(L.LightningDataModule):
    def __init__(self, 
        window_len_s=10, 
        tau_pos_s=10, 
        tau_neg_s=None, 
        same_rec_neg=False, 
        random_state=9, 
        batch_size: int = 64, 
        num_workers=0,
        data_dir='data/hbn_preprocessed',
    ):
        super().__init__()
        self.window_len_s = window_len_s
        self.tau_pos_s = tau_pos_s
        self.tau_neg_s = tau_neg_s
        self.same_rec_neg = same_rec_neg
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.random_state = random_state
        self.data_dir = data_dir

    def preprocess(self, all_ds):
        from sklearn.preprocessing import scale as standard_scale
        os.makedirs(self.data_dir, exist_ok=True)

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
        preprocess(all_ds, preprocessors, save_dir=self.data_dir, overwrite=True, n_jobs=-1)

    def prepare_data(self):
        # create preprocessed data if not exists
        if not os.path.exists(self.data_dir):
            releases = list(range(9,0,-1))
            hbn_datasets = ['ds005514','ds005512','ds005511','ds005510','ds005509','ds005508','ds005507','ds005506','ds005505']
            hbn_release_ds = dict(zip(releases,hbn_datasets))
            selected_releases = [1,2,6]
            selected_tasks = ['RestingState']
            data_path = 'data'
            all_ds = []
            for r in selected_releases:
                if not os.path.exists(f"{data_path}/{hbn_release_ds[r]}"):
                    raise ValueError(f"Data for release {r}-{hbn_release_ds[r]} not found")
                all_ds.append(HBNDataset(hbn_release_ds[r], tasks=selected_tasks, num_workers=-1, preload=False, data_path='data'))

            all_ds = BaseConcatDataset(all_ds)
            self.preprocess(all_ds)

        all_ds = load_concat_dataset(path=self.data_dir, preload=False)

        target_name = 'age'
        for ds in all_ds.datasets:
            ds.target_name = target_name

        fs = all_ds.datasets[0].raw.info['sfreq']
        window_len_samples = int(fs * self.window_len_s)
        window_stride_samples = int(fs * self.window_len_s) # non-overlapping
        self.windows_ds = create_fixed_length_windows(
            all_ds, start_offset_samples=0, stop_offset_samples=None,
            window_size_samples=window_len_samples,
            window_stride_samples=window_stride_samples, drop_last_window=True,
            preload=False)
        
        self.n_channels, self.n_times = self.windows_ds[0][0].shape
        self.sfreq = self.windows_ds.datasets[0].raw.info['sfreq']
        self.tau_pos = int(self.sfreq * self.tau_pos_s)
        self.tau_neg = int(self.sfreq * self.tau_neg_s) if self.tau_neg_s else int(self.sfreq * 2 * self.tau_pos_s)

        subjects = np.unique(self.windows_ds.description['subject'])
        subj_train, subj_test = train_test_split(
            subjects, test_size=0.4, random_state=self.random_state)
        subj_valid, subj_test = train_test_split(
            subj_test, test_size=0.5, random_state=self.random_state)

        self.split_ids = {'train': subj_train, 'valid': subj_valid, 'test': subj_test}
        # get minimum number of samples per dataset
        # subjects = self.windows_ds.get_metadata()['subject'].values
        _, counts = np.unique(subjects, return_counts=True)
        min_sample_per_dataset = np.min(counts)
        self.n_samples_per_dataset = min_sample_per_dataset # this number is a function of window_len_s and recording length

    def setup(self, stage=None):
        if stage == 'fit':
            self.train_ds = RelativePositioningDataset(
                [ds for ds in self.windows_ds.datasets
                if ds.description['subject'] in self.split_ids['train']])
            self.valid_ds = RelativePositioningDataset(
                [ds for ds in self.windows_ds.datasets
                if ds.description['subject'] in self.split_ids['valid']])
            self.valid_ds.return_pair = False
        elif stage == 'test':
            self.test_ds = RelativePositioningDataset(
                [ds for ds in self.windows_ds.datasets
                if ds.description['subject'] in self.split_ids['test']])
            self.test_ds.return_pair = False

    def train_dataloader(self):
        n_examples_train = self.n_samples_per_dataset * len(self.train_ds.datasets)
        train_sampler = DistributedRelativePositioningSampler(
            self.train_ds.get_metadata(), tau_pos=self.tau_pos, tau_neg=self.tau_neg,
            n_examples=n_examples_train, same_rec_neg=self.same_rec_neg, random_state=self.random_state)
        return DataLoader(self.train_ds, sampler=train_sampler, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_ds, sampler=torch.utils.data.distributed.DistributedSampler(self.valid_ds), batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds, sampler=torch.utils.data.distributed.DistributedSampler(self.test_ds), batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        pass

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        pass
    
# define the LightningModule
class LitSSL(L.LightningModule):
    def __init__(self, 
        n_channels, 
        sfreq, 
        input_size_samples, 
        window_len_s, 
        emb_size, 
        dropout=0.5
    ):
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
        print('X train size', x1.element_size() * x1.nelement())
        print('y train size', y.element_size() * y.nelement())
        z1, z2 = self.emb(x1), self.emb(x2)
        z = self.pooling(torch.abs(z1 - z2)).flatten(start_dim=1)

        loss = nn.functional.binary_cross_entropy_with_logits(self.clf(z).flatten(), y)

        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        X, Y, _ = batch
        print('X valid size', X.element_size() * X.nelement())
        print('y valid size', Y.element_size() * Y.nelement())
        z = self.embed(X)
        self.rankme.update(z)

        z = z.float().detach().cpu().numpy()
        Y = Y.float().detach().cpu().numpy()
        from sklearn import linear_model, neural_network
        regr = linear_model.LinearRegression()
        regr, linear_score = train_regressor(regr, z, Y)
        self.log('val_linear_score', linear_score, sync_dist=True)
        regr = neural_network.MLPRegressor(max_iter=1000)
        regr, nn_score = train_regressor(regr, z, Y)
        self.log('val_nn_score', nn_score, sync_dist=True)
        
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
        self.log('val_rankme', self.rankme.compute(), sync_dist=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

def main():
    data_module = RelativePositioningHBNDataModule()
    data_module.prepare_data()

    emb_size = 100
    model = LitSSL(data_module.n_channels, data_module.sfreq, data_module.n_times, data_module.window_len_s, emb_size)

    # # Use the parsed arguments in your program
    # if args.accelerator == 'hpu':
    #     from lightning_habana.pytorch.accelerator import HPUAccelerator
    #     args.accelerator = HPUAccelerator()
    # fast_dev_run=False
    # if args.debug:
    #     fast_dev_run=True
    #     if args.debug == 'pytorch':
    #         from lightning.pytorch.profilers import PyTorchProfiler
    #         from torch.profiler import ProfilerActivity
    #         args.debug = PyTorchProfiler(activities={ProfilerActivity.CPU}, profile_memory=True)
    cli = LightningCLI(model, data_module)
    # trainer = L.Trainer(max_epochs=args.epochs, fast_dev_run=fast_dev_run, accelerator=args.accelerator, devices=args.device, strategy='auto', profiler=args.debug, use_distributed_sampler=False, num_sanity_val_steps=0)

if __name__ == '__main__':
    # from argparse import ArgumentParser
    # parser = ArgumentParser()

    # # Trainer arguments
    # parser.add_argument("--window_len_s", type=int, default=10)
    # parser.add_argument("--tau_pos_s", type=int, default=10)
    # parser.add_argument("--batch_size", type=int, default=64)
    # parser.add_argument("--epochs", type=int, default=1)
    # parser.add_argument("--accelerator", type=str, default='hpu')
    # parser.add_argument("--device", type=str, default='auto')
    # parser.add_argument("--num_workers", type=int, default=4)
    # parser.add_argument("--debug", type=str, default=None)

    # # Parse the user inputs and defaults (returns a argparse.Namespace)
    # args = parser.parse_args()
    # if not args.device == 'auto':
    #     args.device = int(args.device)
    main()
