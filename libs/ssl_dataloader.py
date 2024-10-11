import os

import numpy as np
import mne
import torch
import csv
import pandas as pd
from libs.signalstore_data_utils import SignalstoreHBN
from os import scandir
import xarray as xr
import time
from pathlib import Path
import math
try:
    from importlib import resources as impresources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as impresources

verbose = False
    
class HBNRestBIDSDataset(torch.utils.data.IterableDataset):
    def __init__(self,
            data_dir='/mnt/nemar/openneuro/ds004186', # location of asr cleaned data 
            metadata={
                'file': (impresources.files('libs') / 'subjects.csv'),  # path to subject metadata csv file
                'index_column': 'participant_id',      # index column of the file corresponding to subject name
                'key': 'gender',                       # which metadata we want to use for finetuning
            },                 
            x_params={
                "window": 2,                                          # EEG window length in seconds
                "sfreq": 128,                                         # desired sampling rate
                "subject_per_batch": 10,                              # number of subjects per batch
            },
            random_seed=None):                                               # numpy random seed
        super(HBNRestBIDSDataset).__init__()
        np.random.seed(random_seed)
        self.bidsdir = Path(data_dir)
        self.files = []
        self.M = x_params['sfreq'] * x_params['window']
        self.sfreq = x_params['sfreq']

        # shuffle data
        shuffling_indices = list(range(len(self.files)))
        np.random.shuffle(shuffling_indices)
        # self.metadata_info = metadata
        # self.metadata = self.get_metadata()
        # self.files = self.files[shuffling_indices]

    def __iter__(self):
        if not type(self.bidsdir) is Path:
            self.bidsdir = Path(self.bidsdir)
        for entry in scandir(self.bidsdir):
            if entry.is_dir() and entry.name.startswith('sub-'):
                subject_dir = entry.name
                subject = subject_dir.split('-')[1]
                subject_dir_path = self.bidsdir / subject_dir
                eeg_dir = subject_dir_path / "eeg"

                tasks = ['EC', 'EO']
                runs  = [list(range(1, 6)), list(range(1, 6))]
                for t, task in enumerate(tasks):
                    for run in runs[t]:
                        # get file by name pattern subject_dir*task*run_eeg.set
                        raw_file = eeg_dir / f"{subject_dir}_task-{task}_run-{run}_eeg.set"
                        if os.path.exists(raw_file):
                            self.files.append(raw_file)
                            # load data
                            data = self.preload_raw(raw_file)
                            max_length   = data.shape[-1]

                            # sample windows, rotating through the subjects
                            # to ensure equal contribution among subject per batch
                            indices  = np.arange(0, max_length-self.M, self.M)
                            for idx in indices:
                                if idx < data.shape[-1]-self.M:
                                    yield data[:,idx:idx+self.M] #, self.subjects[i+s]

    def preload_raw(self, raw_file):
        EEG = mne.io.read_raw_eeglab(raw_file, preload=True, verbose='error')
         # bring to common sampling rate
        if EEG.info['sfreq'] != self.sfreq:
            EEG = EEG.resample(self.sfreq)
        mat_data = EEG.get_data()

        if len(mat_data.shape) > 2:
            raise ValueError('Expect raw data to be CxT dimension')
        return mat_data


class HBNRestDataset(torch.utils.data.IterableDataset):
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
                "subject_per_batch": 10,                              # number of subjects per batch
            },
            is_test=False,                                            # use (folds-1 or 1 fold) if n_cv != None
            random_seed=None):                                               # numpy random seed
        super(HBNRestDataset).__init__()
        np.random.seed(random_seed)
        self.basedir = data_dir
        self.files = np.array([i for i in os.listdir(self.basedir) if i.split('.')[-1] == 'set'])
        self.subjects = np.array([i.split('_')[0] for i in os.listdir(self.basedir) if i.split('.')[-1] == 'set'])
        self.M = x_params['sfreq'] * x_params['window']
        self.subject_per_batch = x_params['subject_per_batch']

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
        # set up multi-processing
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = 0
            iter_end = len(self.files)
        else:  # in a worker process
            # split workload
            per_worker = int(math.ceil(len(self.files) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.files))

        for i in range(iter_start, iter_end, self.subject_per_batch):
            # load data
            subject_data = [self.preload_raw(self.files[i+s]) for s in range(self.subject_per_batch)]
            max_length   = max([data.shape[-1] for data in subject_data])

            # sample windows, rotating through the subjects
            # to ensure equal contribution among subject per batch
            indices  = np.arange(0, max_length-self.M, self.M)
            for idx in indices:
                for s in range(self.subject_per_batch):
                    data = subject_data[s]
                    if idx < data.shape[-1]-self.M:
                        yield data[:,idx:idx+self.M] #, self.subjects[i+s]


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
        EEG = mne.io.read_raw_eeglab(os.path.join(self.basedir, raw_file), preload=True, verbose='error')
        mat_data = EEG.get_data()

        if len(mat_data.shape) > 2:
            raise ValueError('Expect raw data to be CxT dimension')
        return mat_data


class HBNSignalstoreDataset(torch.utils.data.IterableDataset):
    def __init__(self,
            metadata={
                'filepath': '/home/dung/subjects.csv', # path to subject metadata csv file
                'index_column': 'participant_id',      # index column of the file corresponding to subject name
                'key': 'gender',                       # which metadata we want to use for finetuning
            },                 
            dataset_name='healthy-brain-network',      # signalstore dataset name
            subjects:list=[],                                       # subjects to use, default to all
            n_subjects=-1,                                          # number of subjects to pick, default all
            task_params={
                "window": 2,                                          # EEG window length in seconds
                "sfreq": 128,                                         # sampling rate
                "task": "EC",                                         # list of task name(s)
            },
            dbconnectionstring:str='',
            is_test=False,                                            # use (folds-1 or 1 fold) if n_cv != None
            random_seed=9):                                               # numpy random seed
        self.seed = random_seed
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        self.signalstore = SignalstoreHBN(dataset_name, dbconnectionstring=dbconnectionstring)
        self.metadata_info= metadata
        self.nsubjects = n_subjects
        self.task_params = task_params
        self.M = task_params['sfreq'] * task_params['window']
        
        query = {'task': self.task_params['task']} # query = {'task': {'$in': self.task_params['task']}} 
        self.records = self.signalstore.query_data(query)


    def __iter__(self):
        # set up multi-processing
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = 0
            iter_end = len(self.records)
        else:  # in a worker process
            # split workload
            per_worker = int(math.ceil(len(self.records) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.records))

        for i in range(iter_start, iter_end):
            record = self.records[i]
            data = self.signalstore.query_data({'schema_ref': record['schema_ref'], 'data_name': record['data_name']}, get_data=True)[0]

            # sample windows, rotating through the subjects
            # to ensure equal contribution among subject per batch
            indices  = np.arange(0, data.shape[-1]-self.M, self.M)
            for idx in indices:
                yield data[:,idx:idx+self.M].to_numpy() #, self.subjects[i+s]


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


if __name__ == "__main__":
    dataset = HBNSignalstoreDataset(
        task_params={
            "window": 2,
            "sfreq": 128,
            "task": "EC",
        },
    )
    print(list(torch.utils.data.DataLoader(dataset)))