import os

import numpy as np
import mne
import torch
from tqdm import tqdm
import hashlib
from pathlib import Path
import csv
# from abc import ABC, abstractmethod
import pickle
from joblib import Parallel, delayed
import pandas as pd
from libs.signalstore_data_utils import SignalstoreHBN
import xarray as xr
import time
try:
    from importlib import resources as impresources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as impresources

verbose = False
    
class MaskedContrastiveLearningDataset(torch.utils.data.IterableDataset):
    def __init__(self,
            data_dir='/mnt/nemar/child-mind-rest', # location of asr cleaned data 
            metadata={
                'file': (impresources.files('libs') / 'subjects.csv'),  # path to subject metadata csv file
                'index_column': 'participant_id',      # index column of the file corresponding to subject name
                'key': 'gender',                       # which metadata we want to use for finetuning
            },                 
            subjects:list=None,                                       # subjects to use, default to all
            n_subjects=None,                                          # number of subjects to pick, default all
            x_params={
                "window": 2,                                          # EEG window length in seconds
                "sfreq": 128,                                         # sampling rate
            },
            is_test=False,                                            # use (folds-1 or 1 fold) if n_cv != None
            random_seed=None):                                               # numpy random seed
        np.random.seed(random_seed)
        self.basedir = data_dir
        self.files = np.array([i for i in os.listdir(self.basedir) if i.split('.')[-1] == 'set'])
        self.subjects = np.array([i.split('_')[0] for i in os.listdir(self.basedir) if i.split('.')[-1] == 'set'])
        self.M = x_params['sfreq'] * x_params['window']
        if subjects != None:
            subjects = [i for i in subjects if i in self.subjects]
            selected_indices = [subjects.index(i) for i in subjects if i in self.subjects]
            self.subjects = np.array(subjects)[selected_indices]
            self.files = self.files[selected_indices]
            if len(subjects) - len(self.subjects) > 0:
                print("Warning: unknown keys present in user specified subjects")
        shuffling_indices = list(range(len(self.subjects)))
        np.random.shuffle(shuffling_indices)
        self.metadata_info = metadata
        self.metadata = self.get_metadata()
        self.subjects = self.subjects[shuffling_indices]
        self.files = self.files[shuffling_indices]

        n_subjects = n_subjects if n_subjects is not None else len(self.subjects)
        if n_subjects > len(self.subjects):
            print("Warning: n_subjects cannot be larger than subjects")
        self.subjects = self.subjects[:n_subjects]
        self.files = self.files[:n_subjects]


        # Split Train-Test TODO
        # self.is_test = is_test
        # if self.n_cv is not None:
        #     split_size = int(n_subjects / self.n_cv[1])
        #     if not self.is_test:
        #         self.subjects = self.subjects[:self.n_cv[0]*split_size] + \
        #                         self.subjects[(self.n_cv[0]+1)*split_size:]
        #     else:
        #         self.subjects = self.subjects[self.n_cv[0]*split_size:(self.n_cv[0]+1)*split_size]

    def __iter__(self):
        for i in range(len(self.files)):
            raw_file = self.files[i]
            data = self.preload_raw(raw_file) 
            y = self.metadata[i]
            indices = np.arange(0, data.shape[-1]-self.M, self.M)
            for idx in indices:
                yield data[:,idx:idx+self.M], y

    '''
    Extract metadata of samples
    '''
    def get_metadata(self):
        key = self.metadata_info['key']
        with open(self.metadata_info['file'], 'rb') as f:  # or "rt" as text file with universal newlines
            subj_info = pd.read_csv(f, index_col=self.metadata_info['index_column']) # master sheet containing all subjects
        # subj_info = pd.read_csv(self.metadata_info['filepath'], index_col=self.metadata_info['index_column']) # master sheet containing all subjects
        if key not in subj_info:
            print('Metadata key not found')
            return 
        else:
            metadata = [subj_info[key][subj] for subj in self.subjects]
            return metadata

    def preload_raw(self, raw_file):
        EEG = mne.io.read_raw_eeglab(os.path.join(raw_file), preload=True)
        mat_data = EEG.get_data()

        if len(mat_data.shape) > 2:
            raise ValueError('Expect raw data to be CxT dimension')
        return mat_data


class MaskedContrastiveLearningSignalstoreDataset(torch.utils.data.Dataset):
    def __init__(self,
            metadata={
                'filepath': '/home/dung/subjects.csv', # path to subject metadata csv file
                'index_column': 'participant_id',      # index column of the file corresponding to subject name
                'key': 'gender',                       # which metadata we want to use for finetuning
            },                 
            dataset_name='healthy_brain_network',      # signalstore dataset name
            subjects:list=[],                                       # subjects to use, default to all
            n_subjects=-1,                                          # number of subjects to pick, default all
            task_params={
                "window": 2,                                          # EEG window length in seconds
                "task": "EC",                                         # list of task name(s)
            },
            dbconnectionstring:str='',
            is_test=False,                                            # use (folds-1 or 1 fold) if n_cv != None
            random_seed=9):                                               # numpy random seed
        np.random.seed(random_seed)
        self.signalstore = SignalstoreHBN(dataset_name, dbconnectionstring=dbconnectionstring)
        self.metadata_info= metadata
        self.nsubjects = n_subjects
        self.task_params = task_params
        self.subjects = subjects if subjects else self.__get_subjects(is_test) 

        self.seed = random_seed
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        self.data = None
        self.start_time = time.time()

        '''
        TODO can we do all this lazily? As in all these operations are done on the metadata level and actual data will be retrieved only when pytorch request batch data.
             can the indexing be computed by time dimension metadata without loading the data?
        '''
        self.__get_data()
        self.start_time = time.time()
        self.__process_data()
        self.start_time = time.time()

        # enforcing data order for uniform array computation
        self.data = self.data.transpose('sample', 'channel', 'time')

        # set metadata
        self.sfreq = self.data.attrs['sampling_frequency']
        print('Data shape:', self.data.shape)

    def __len__(self):
        return self.data.sample.shape

    def __getitem__(self, idx):
        sample = self.data.isel(sample=idx)
        return sample.to_numpy(), sample.attrs
    
    def __get_subjects(self, is_test):
        '''
        Get list of subjects to be used in the dataset, depending on whether it's test or traing data
        '''
        basedir = "/mnt/nemar/child-mind-rest"
        
        test_prob = 0.3
        all_subjects = np.array([i.split('_')[0] for i in os.listdir(basedir) if i.split('.')[-1] == 'set'])
        test_subjects_file = impresources.files('libs').joinpath('data').joinpath('test_subjects.csv')
        if test_subjects_file.exists():
            with test_subjects_file.open() as f:
                reader = csv.reader(f)
                test_subjects = next(reader)
        else:
            test_subjects = all_subjects[:int(len(all_subjects)*test_prob)]
            with test_subjects_file.open('w') as out:
                writer = csv.writer(out)
                writer.writerow(test_subjects)
        if is_test:
            subjects = test_subjects
        else:
            subjects = list(set(all_subjects)-set(test_subjects))
        
        random_idx = np.arange(len(subjects))
        np.random.shuffle(random_idx)
        subjects = list(np.array(subjects)[random_idx])

        return subjects


    # def __get_batch(self):
    def __get_data(self):
        '''
        '''
        print('Getting subject data for task %s...' % (self.task_params['task']))
        ds = []
        subj_count = 0

        for subj in self.subjects:
            if self.nsubjects > 0 and subj_count == self.nsubjects:
                break
            print('Subject:', subj)
            print('Querying...')
            query = {'subject': subj, 'task': {'$in': self.task_params['task']}} #if self.task_params['task'] else {'subject': subj}
            recordings = self.signalstore.query_data(query)
            print('Took %s second(s)' % (time.time() - self.start_time))
            self.start_time = time.time()

            print('Retrieving raw data...')
            for r in recordings:
                record = self.signalstore.query_data({'schema_ref': r['schema_ref'], 'data_name': r['data_name']}, get_data=True, validate=False)[0]
                ds.append(record)
            print('Took %s second(s)' % (time.time() - self.start_time))
            subj_count += 1
        
        self.data = xr.merge(ds)


    def __process_data(self):
        print('Processing data...')
        window = self.task_params['window']
        ds = self.data

        # segment data into EEG windows
        ds_coarsen = ds.coarsen(time=ds.attrs['sampling_frequency']*window, boundary='trim').construct(time=('window', 'time'), keep_attrs=True)
        ds2 = ds_coarsen.to_dataarray()
        ds3 = ds2.stack(sample=("variable", "window"))
        self.data = ds3

        # shuffle data
        print('\tShuffling...')
        random_idx = np.arange(ds3.sample.shape[0])
        np.random.shuffle(random_idx)
        ds3 = ds3.isel(sample=random_idx)
        print('Took %s second(s)' % (time.time() - self.start_time))
        self.start_time = time.time()

    def get_metadata(self, key):
        '''
        Extract metadata of samples
        '''
        subj_info = pd.read_csv(self.metadata_info['filepath'], index_col=self.metadata_info['index_column']) # master sheet containing all subjects
        if key not in subj_info:
            print('Metadata key not found')
            return 
        else:
            metadata = [subj_info[key][subj] for subj in self.subjects]
            return metadata

