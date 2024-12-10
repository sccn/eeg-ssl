import os
import sys 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from concurrent.futures import ThreadPoolExecutor, as_completed
from joblib import Parallel, delayed
import numpy as np
import mne
import torch
from sklearn import preprocessing
import csv
import pandas as pd
# from libs.signalstore_data_utils import SignalstoreHBN
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

class BIDSDataset(torch.utils.data.IterableDataset):
    ALLOWED_FILE_FORMAT = ['eeglab', 'brainvision']
    RAW_EXTENSION = {
        'eeglab': '.set',
        'brainvision': '.vhdr'
    }
    X_PARAMS = {
        "window": 2,                                          # EEG window length in seconds
        "sfreq": 128,                                         # desired sampling rate
        "preprocess": False,                                  # whether preprocess data
    }
    def __init__(self,
            data_dir=None,                            # location of asr cleaned data 
            raw_format='eeglab',                      # format of raw data
            metadata={
                'file': (impresources.files('libs') / 'subjects.csv'),  # path to subject metadata csv file
                'index_column': 'participant_id',      # index column of the file corresponding to subject name
                'key': 'gender',                       # which metadata we want to use for finetuning
            },                 
            x_params=X_PARAMS,                         # parameters for data preprocessing
            random_seed=0):                            # numpy random seed
        super().__init__()
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        if data_dir is None:
            raise ValueError('data_dir must be specified')
        self.bidsdir = Path(data_dir)

        if raw_format.lower() not in self.ALLOWED_FILE_FORMAT:
            raise ValueError('raw_format must be one of {}'.format(self.ALLOWED_FILE_FORMAT))
        self.raw_format = raw_format.lower()

        # get all .set files in the bids directory
        temp_dir = (Path().resolve() / 'data')
        if not os.path.exists(temp_dir):
            os.mkdir(temp_dir)
        if not os.path.exists(temp_dir / 'files.npy'):
            self.files = self.get_files_with_extension_parallel(self.bidsdir, extension=self.RAW_EXTENSION[self.raw_format])
            np.save(temp_dir / 'files.npy', self.files)
        else:
            self.files = np.load(temp_dir / 'files.npy', allow_pickle=True)

        self.M = x_params['sfreq'] * x_params['window']
        self.preprocess = False

        for name, value in x_params.items():
            setattr(self, name, value)

        self.sfreq = x_params['sfreq']

        self._shuffle # in place shuffling

    def _shuffle(self):
        print('Shuffling data')
        shuffling_indices = list(range(len(self.files)))
        np.random.shuffle(shuffling_indices)
        self.files = self.files[shuffling_indices]
        # self.metadata_info = metadata
        # self.metadata = self.get_metadata()

    def scan_directory(self, directory, extension):
        result_files = []
        directory_to_ignore = ['.git']
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.endswith(extension):
                    print('Adding ', entry.path)
                    result_files.append(entry.path)
                elif entry.is_dir():
                    # check that entry path doesn't contain any name in ignore list
                    if not any(name in entry.name for name in directory_to_ignore):
                        result_files.append(entry.path)  # Add directory to scan later
        return result_files

    def get_files_with_extension_parallel(self, directory, extension='.set', max_workers=-1):
        result_files = []
        dirs_to_scan = [directory]

        # Use joblib.Parallel and delayed to parallelize directory scanning
        while dirs_to_scan:
            print(f"Scanning {len(dirs_to_scan)} directories...", dirs_to_scan)
            # Run the scan_directory function in parallel across directories
            results = Parallel(n_jobs=max_workers, prefer="threads", verbose=1)(
                delayed(self.scan_directory)(d, extension) for d in dirs_to_scan
            )
            
            # Reset the directories to scan and process the results
            dirs_to_scan = []
            for res in results:
                for path in res:
                    if os.path.isdir(path):
                        dirs_to_scan.append(path)  # Queue up subdirectories to scan
                    else:
                        result_files.append(path)  # Add files to the final result
            print(f"Current number of files: {len(result_files)}")

        return result_files

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
            print(f'worker_id: {worker_id}, iter_start: {iter_start}, iter_end: {iter_end}\n')
        for i in range(iter_start, iter_end):
            raw_file = self.files[i]
            if os.path.exists(raw_file):
                # load data
                data = self.load_and_preprocess_raw(raw_file)
                max_length   = data.shape[-1]

                # sample windows, rotating through the subjects
                # to ensure equal contribution among subject per batch
                indices  = np.arange(0, max_length-self.M, self.M)
                for idx in indices:
                    if idx < data.shape[-1]-self.M:
                        yield data[:,idx:idx+self.M] #, self.subjects[i+s]

    def load_and_preprocess_raw(self, raw_file):
        EEG = mne.io.read_raw_eeglab(raw_file, preload=True, verbose='error')
        
        if self.preprocess:
            # highpass filter
            EEG = EEG.filter(l_freq=0.25, h_freq=25, verbose=False)
            # remove 60Hz line noise
            EEG = EEG.notch_filter(freqs=(60), verbose=False)
            # bring to common sampling rate

        if EEG.info['sfreq'] != self.sfreq:
            EEG = EEG.resample(self.sfreq)

        mat_data = EEG.get_data()
        mat_data = mat_data[0:128, :] # remove Cz reference chan

        # normalize data to zero mean and unit variance
        scalar = preprocessing.StandardScaler()
        mat_data = scalar.fit_transform(mat_data.T).T # scalar normalize for each feature and expects shape data x features

        if len(mat_data.shape) > 2:
            raise ValueError('Expect raw data to be CxT dimension')
        return mat_data
    
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
                "preprocess": False,                                  # whether preprocess data
            },
            random_seed=0):                                               # numpy random seed
        super(HBNRestBIDSDataset).__init__()
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        self.bidsdir = Path(data_dir)
        # get all .set files in the bids directory
        temp_dir = (Path().resolve() / 'data')
        if not os.path.exists(temp_dir):
            os.mkdir(temp_dir)
        if not os.path.exists(temp_dir / 'files.npy'):
            self.files = self.get_files_with_extension_parallel(self.bidsdir, extension='.set')
            np.save(temp_dir / 'files.npy', self.files)
        else:
            self.files = np.load(temp_dir / 'files.npy', allow_pickle=True)

        self.M = x_params['sfreq'] * x_params['window']
        self.preprocess = False

        for name, value in x_params.items():
            setattr(self, name, value)

        self.sfreq = x_params['sfreq']

        self._shuffle # in plac

    def _shuffle(self):
        # shuffle data
        shuffling_indices = list(range(len(self.files)))
        np.random.shuffle(shuffling_indices)
        self.files = self.files[shuffling_indices]
        # self.metadata_info = metadata
        # self.metadata = self.get_metadata()

    def scan_directory(self, directory, extension):
        result_files = []
        directory_to_ignore = ['.git']
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.endswith(extension):
                    print('Adding ', entry.path)
                    result_files.append(entry.path)
                elif entry.is_dir():
                    # check that entry path doesn't contain any name in ignore list
                    if not any(name in entry.name for name in directory_to_ignore):
                        result_files.append(entry.path)  # Add directory to scan later
        return result_files

    def get_files_with_extension_parallel(self, directory, extension='.set', max_workers=-1):
        result_files = []
        dirs_to_scan = [directory]

        # Use joblib.Parallel and delayed to parallelize directory scanning
        while dirs_to_scan:
            print(f"Scanning {len(dirs_to_scan)} directories...", dirs_to_scan)
            # Run the scan_directory function in parallel across directories
            results = Parallel(n_jobs=max_workers, prefer="threads", verbose=1)(
                delayed(self.scan_directory)(d, extension) for d in dirs_to_scan
            )
            
            # Reset the directories to scan and process the results
            dirs_to_scan = []
            for res in results:
                for path in res:
                    if os.path.isdir(path):
                        dirs_to_scan.append(path)  # Queue up subdirectories to scan
                    else:
                        result_files.append(path)  # Add files to the final result
            print(f"Current number of files: {len(result_files)}")

        return result_files

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
            print('worker_id', worker_id, 'iter_start', iter_start, 'iter_end', iter_end, '\n')
        for i in range(iter_start, iter_end):
            raw_file = self.files[i]
            if os.path.exists(raw_file):
                # load data
                data = self.load_and_preprocess_raw(raw_file)
                max_length   = data.shape[-1]

                # sample windows, rotating through the subjects
                # to ensure equal contribution among subject per batch
                indices  = np.arange(0, max_length-self.M, self.M)
                for idx in indices:
                    if idx < data.shape[-1]-self.M:
                        yield data[:,idx:idx+self.M] #, self.subjects[i+s]

    def load_and_preprocess_raw(self, raw_file):
        EEG = mne.io.read_raw_eeglab(raw_file, preload=True, verbose='error')
        
        if self.preprocess:
            # highpass filter
            EEG = EEG.filter(l_freq=0.25, h_freq=25, verbose=False)
            # remove 60Hz line noise
            EEG = EEG.notch_filter(freqs=(60), verbose=False)
            # bring to common sampling rate

        if EEG.info['sfreq'] != self.sfreq:
            EEG = EEG.resample(self.sfreq)

        mat_data = EEG.get_data()
        mat_data = mat_data[0:128, :] # remove Cz reference chan

        # normalize data to zero mean and unit variance
        scalar = preprocessing.StandardScaler()
        mat_data = scalar.fit_transform(mat_data.T).T # scalar normalize for each feature and expects shape data x features

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
    dataset = HBNRestBIDSDataset(
        data_dir = "/mnt/nemar/openneuro/ds004186", # ds004186 ds005510
        x_params = {
            'sfreq': 128,
            'window': 20,
            'preprocess': False,
        },
    )