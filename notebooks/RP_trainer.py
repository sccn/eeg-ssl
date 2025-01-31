import sys
sys.path.insert(0, '../')
from libs.ssl_dataloader import *
from libs.ssl_model import *
from libs.ssl_utils import *
from libs.ssl_utils import DistributedRelativePositioningSampler
from libs.eeg_utils import *
from braindecode.preprocessing import (
    preprocess, Preprocessor, create_fixed_length_windows)
from braindecode.datasets import BaseConcatDataset
from braindecode.datautil import load_concat_dataset
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
        overwrite_preprocessed=False,
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
        self.overwrite_preprocessed = overwrite_preprocessed

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
        if not os.path.exists(self.data_dir) or self.overwrite_preprocessed:
            os.makedirs(self.data_dir, exist_ok=True)
            releases = list(range(9,0,-1))
            hbn_datasets = ['ds005514','ds005512','ds005511','ds005510','ds005509','ds005508','ds005507','ds005506','ds005505']
            hbn_release_ds = dict(zip(releases,hbn_datasets))
            selected_releases = [1,3,6]
            selected_tasks = ['RestingState']
            data_path = 'data'
            all_ds = []
            for r in selected_releases:
                if not os.path.exists(f"{data_path}/{hbn_release_ds[r]}"):
                    raise ValueError(f"Data for release {r}-{hbn_release_ds[r]} not found")
                all_ds.append(HBNDataset(hbn_release_ds[r], tasks=selected_tasks, num_workers=-1, preload=False, data_path='data'))

            all_ds = BaseConcatDataset(all_ds)
            self.preprocess(all_ds)

    def setup(self, stage=None):
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

    def configure_optimizers(self):
            optimizer = optim.Adam(self.parameters(), lr=1e-3)
            return optimizer 

def main():
    cli = LightningCLI(LitSSL, RelativePositioningHBNDataModule)

if __name__ == '__main__':
    main()
