import os
from pathlib import Path
import re
import warnings
import json
from typing import Any
from joblib import Parallel, delayed
import mne
import scipy
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from braindecode.datasets import BaseDataset, BaseConcatDataset
from braindecode.preprocessing import (
    preprocess, Preprocessor, create_fixed_length_windows)
from braindecode.datautil import load_concat_dataset
import torch
from torch import optim
from torch.utils.data import DataLoader
import torch.distributed as dist
from .ssl_task import *
from eegdash import EEGDashDataset
import lightning as L

class SSLHBNDataModule(L.LightningDataModule):
    def __init__(self, 
        ssl_task: SSLTask = RelativePositioning,
        window_len_s=10, 
        batch_size: int = 64, 
        num_workers=0,
        data_dir='/mnt/nemar/openneuro',
        cache_dir='data',
        datasets:list[str]=None, # traing datasets
        target_label='age',
        overwrite_preprocessed=False,
        mapping=None,
        val_release='ds005505',
        test_release='ds005510',
        use_ssl_sampler_for_val=False,
        train_percent=1.0,
    ):
        super().__init__()
        self.ssl_task = ssl_task
        self.window_len_s = window_len_s
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else self.data_dir
        self.overwrite_preprocessed = overwrite_preprocessed
        HBN_DSNUMBERS = ['ds005512','ds005510','ds005509','ds005508','ds005507','ds005505']
        self.datasets = datasets if datasets is not None else HBN_DSNUMBERS
        self.bad_subjects = ['NDARBA381JGH', 'NDARUJ292JXV', 'NDARVN772GLC', 'NDARTD794NKQ', 'NDARBX830ZD4', 'NDARHZ923PAH', 'NDARJP304NK1', 'NDARME789TD2', 'NDARUA442ZVF', 'NDARTY128YLU', 'NDARDW550GU6','NDARLD243KRE']
        self.target_label = target_label
        self.mapping = mapping
        self.val_release = val_release
        self.test_release = test_release
        self.use_ssl_sampler_for_val = use_ssl_sampler_for_val
        self.train_percent = train_percent
        self.save_hyperparameters()

    def prepare_data(self):
        # create preprocessed data if not exists
        print(f"Train releases: {self.datasets}")
        print(f"Validation release: {self.val_release}")
        print(f"Test release: {self.test_release}")

        assert self.val_release not in self.datasets, "Validation release should not be in training releases"
        assert self.test_release not in self.datasets, "Test release should not be in training releases"

        for dsnumber in [*self.datasets, self.val_release, self.test_release]:
            savedir = self.cache_dir / f'{dsnumber}_preprocessed'
            if not os.path.exists(savedir) or self.overwrite_preprocessed:
                ds = EEGDashDataset({'dataset': dsnumber, 'task': 'RestingState'}, cache_dir=self.data_dir, description_fields=['subject', 'session', 'run', 'task', 'age', 'gender', 'sex', 'p_factor', 'externalizing', 'internalizing', 'attention'])
                ds = BaseConcatDataset([d for d in ds.datasets if d.description['subject'] not in self.bad_subjects])
                ds = self.preprocess(ds, savedir)
        
    
    def window_scale(self, data):
        '''
        Legacy reference
        Custom scaling using window statistics ignoring Cz reference
        '''
        assert data.ndim == 3, "Data should be 3D"
        assert data.shape[1] == 129, "Data should have 129 channels"
        assert torch.allclose(data[:,-1,:], torch.tensor(0, dtype=data.dtype)), "Last channel should be Cz reference"
        data_noCz = data[:,:-1,:]  # remove Cz reference
        assert data_noCz.shape[1] == 128, "Data should have 128 channels after removing Cz reference"
        data_mean = torch.mean(data_noCz, dim=(1,2), keepdim=True)  # mean over F
        data_std = torch.std(data_noCz, dim=(1,2), keepdim=True)  # std over F
        # standard scale to 0 mean and 1 std using statistics of the entire window
        data = (data - data_mean) / data_std # normalize preserving batch dim
        return data
            
    def preprocess(self, ds, savedir):
        from sklearn.preprocessing import scale as standard_scale
        os.makedirs(savedir, exist_ok=True)

        def global_norm(data, method='standard'):
            if method == 'standard':
                center = np.mean(data)  
                variance = np.std(data)  
            elif method == 'robust':
                center = np.median(data)  
                variance = np.percentile(data, 75) - np.percentile(data, 25)
            # standard scale to 0 mean and 1 std using statistics of the entire recording
            data = (data - center) / variance # normalize preserving batch dim
            return data
        
        sampling_rate = 250 # resample to follow the tutorial sampling rate
        # Factor to convert from uV to V
        factor = 1e6
        channels = ds.datasets[0].raw.info['ch_names']
        preprocessors = [
            Preprocessor('set_channel_types', mapping=dict(zip(channels, ['eeg']*len(channels)))),
            Preprocessor('notch_filter', freqs=(60, 120)),    
            Preprocessor('filter', l_freq=0.1, h_freq=59),
            Preprocessor('resample', sfreq=sampling_rate),
            Preprocessor('set_eeg_reference', ref_channels=['Cz']),
            # Preprocessor('crop', tmin=10),  # crop first 10 seconds as begining of noise recording
            Preprocessor('drop_channels', ch_names=['Cz']),  # discard Cz
            Preprocessor(global_norm, method='robust'),  
            # Preprocessor(standard_scale, channel_wise=True), # normalization for deep learning
        ]
        # Transform the data
        preprocess(ds, preprocessors, save_dir=savedir, overwrite=True, n_jobs=1)

        return ds

    def get_and_filter_dataset(self, dataset_type='train'):
        if dataset_type == 'train':
            all_ds = BaseConcatDataset([load_concat_dataset(path=self.cache_dir / f'{dsnumber}_preprocessed', preload=False) for dsnumber in self.datasets if dsnumber != self.val_release and dsnumber != self.test_release])
            if self.train_percent < 1.0:
                subjects = np.unique(all_ds.description['subject'])
                subj_train, _ = train_test_split(
                    subjects, train_size=self.train_percent)
                all_ds = BaseConcatDataset([ds for ds in all_ds.datasets if ds.description['subject'] in subj_train])
        elif dataset_type == 'valid':
            all_ds = BaseConcatDataset([load_concat_dataset(path=self.cache_dir / f'{self.val_release}_preprocessed', preload=False)])
        elif dataset_type == 'test':
            all_ds = BaseConcatDataset([load_concat_dataset(path=self.cache_dir / f'{self.test_release}_preprocessed', preload=False)])

        filtered_ds = []

        # check target label validity
        if self.target_label not in all_ds.description.columns:
            raise ValueError(f"Target label {self.target_label} not found in dataset description")
        for ds in all_ds.datasets:
            # filter nan target label
            if not (pd.isna(ds.description[self.target_label]) or ds.description['subject'] in self.bad_subjects):
                if len(ds.raw.ch_names) < 128:
                    raise ValueError(f"Dataset {ds.description['subject']} has less than 128 channels")
                ds.target_name = [self.target_label, 'subject']
                filtered_ds.append(ds)

        all_ds = BaseConcatDataset(filtered_ds)

        # Extract windows
        fs = all_ds.datasets[0].raw.info['sfreq']
        window_len_samples = int(fs * self.window_len_s)
        window_stride_samples = int(fs * self.window_len_s) # non-overlapping
        windows_ds = create_fixed_length_windows(
            all_ds, start_offset_samples=0, stop_offset_samples=None,
            window_size_samples=window_len_samples,
            window_stride_samples=window_stride_samples, drop_last_window=True,
            preload=False, mapping=self.mapping)
        

        return windows_ds

    def setup(self, stage=None):
        if stage == 'fit':
            # use all datasets for training
            train_ds = self.get_and_filter_dataset('train')
            valid_ds = self.get_and_filter_dataset('valid')

            assert set(train_ds.description['subject']).intersection(set(valid_ds.description['subject'])) == set(), "Train and valid datasets should not overlap"

            self.train_ds = train_ds
            self.valid_ds = valid_ds
        elif stage == 'validate':
            self.valid_ds = self.get_and_filter_dataset('valid')
        elif stage == 'test':
            self.test_ds = self.get_and_filter_dataset('test')

        if self.target_label == 'sex':
            # if self has train_ds
            if hasattr(self, 'train_ds'):
                # Get balanced indices for male and female subjects and create a balanced dataset
                male_subjects   = self.train_ds.description['subject'][self.train_ds.description['sex'] == 'M']
                female_subjects = self.train_ds.description['subject'][self.train_ds.description['sex'] == 'F']
                n_samples = min(len(male_subjects), len(female_subjects))
                train_subj = np.concatenate([male_subjects[:n_samples], female_subjects[:n_samples]])
                train_gender = ['M'] * n_samples + ['F'] * n_samples
                # train_subj, val_subj, train_gender, val_gender = train_test_split(balanced_subjects, balanced_gender, train_size=1, stratify=balanced_gender)

                # Create datasets
                self.train_ds = BaseConcatDataset([ds for ds in self.train_ds.datasets if ds.description.subject in train_subj])

                # Check the balance of the dataset
                assert len(train_subj) == len(train_gender)
                print(f"Number of subjects in balanced dataset: {len(train_subj)}")
                print(f"Gender distribution in balanced dataset: {np.unique(train_gender, return_counts=True)}")

            if hasattr(self, 'valid_ds'):
                # Get balanced indices for male and female subjects and create a balanced dataset
                male_subjects   = self.valid_ds.description['subject'][self.valid_ds.description['sex'] == 'M']
                female_subjects = self.valid_ds.description['subject'][self.valid_ds.description['sex'] == 'F']
                n_samples = min(len(male_subjects), len(female_subjects))
                val_subj = np.concatenate([male_subjects[:n_samples], female_subjects[:n_samples]])
                val_gender = ['M'] * n_samples + ['F'] * n_samples
                # train_subj, val_subj, train_gender, val_gender = train_test_split(balanced_subjects, balanced_gender, train_size=1, stratify=balanced_gender)

                # Create datasets
                self.valid_ds = BaseConcatDataset([ds for ds in self.valid_ds.datasets if ds.description.subject in val_subj])

                # Check the balance of the dataset
                assert len(val_subj) == len(val_gender)
                print(f"Number of subjects in balanced dataset: {len(val_subj)}")
                print(f"Gender distribution in balanced dataset: {np.unique(val_gender, return_counts=True)}")

            if hasattr(self, 'test_ds'):
                # Get balanced indices for male and female subjects and create a balanced dataset
                male_subjects   = self.test_ds.description['subject'][self.test_ds.description['sex'] == 'M']
                female_subjects = self.test_ds.description['subject'][self.test_ds.description['sex'] == 'F']
                n_samples = min(len(male_subjects), len(female_subjects))
                test_subj = np.concatenate([male_subjects[:n_samples], female_subjects[:n_samples]])
                test_gender = ['M'] * n_samples + ['F'] * n_samples

                # Create datasets
                self.test_ds = BaseConcatDataset([ds for ds in self.test_ds.datasets if ds.description.subject in test_subj])

                # Check the balance of the dataset
                assert len(test_subj) == len(test_gender)
                print(f"Number of subjects in balanced dataset: {len(test_subj)}")
                print(f"Gender distribution in balanced dataset: {np.unique(test_gender, return_counts=True)}")


    def train_dataloader(self):
        train_sampler = self.ssl_task.sampler(self.train_ds)
        self.train_ds = self.ssl_task.dataset(self.train_ds.datasets)
        shuffle = True if train_sampler is None else False
        if train_sampler:
            print(f"Using {type(train_sampler).__name__} sampler with shuffle {shuffle} for training")
            dataloader = DataLoader(self.train_ds, sampler=train_sampler, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle)
        else:
            dataloader = DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle)
        if not dist.is_initialized():
            print(f"Number of datasets: {len(self.train_ds.datasets)}")
            if train_sampler is None:
                print(f"Number of examples: {len(self.train_ds)}")
            else:
                print(f"Number of examples: {train_sampler.n_examples}")
        else:
            # print(f"Number of datasets for rank {dist.get_rank()}: {dataloader.sampler.n_recordings}")
            print(f"Number of batches for rank {dist.get_rank()}: {len(dataloader)}")
            print(f"Number of examples for rank {dist.get_rank()}: {len(dataloader.sampler) // dist.get_world_size()}")
        return dataloader

    def val_dataloader(self):
        val_sampler = self.ssl_task.sampler(self.valid_ds)
        self.valid_ds = self.ssl_task.dataset(self.valid_ds.datasets)
        if self.use_ssl_sampler_for_val and val_sampler is not None:
            print(f"Using {type(val_sampler).__name__} sampler for validation")
            return DataLoader(self.valid_ds, sampler=val_sampler, batch_size=self.batch_size, num_workers=self.num_workers)
        else:
            return DataLoader(self.valid_ds, batch_size=self.batch_size,  num_workers=self.num_workers)

    def test_dataloader(self):
        test_sampler = self.ssl_task.sampler(self.test_ds)
        self.test_ds = self.ssl_task.dataset(self.test_ds.datasets)
        if self.use_ssl_sampler_for_val and test_sampler is not None:
            print(f"Using {type(test_sampler).__name__} sampler for validation")
            return DataLoader(self.test_ds, sampler=test_sampler, batch_size=self.batch_size, num_workers=self.num_workers)
        else:
            return DataLoader(self.test_ds, batch_size=self.batch_size, collate_fn=self.custom_collate_fn, num_workers=self.num_workers)

    def predict_dataloader(self):
        pass

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        pass

    