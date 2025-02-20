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

import lightning as L

class SSLHBNDataModule(L.LightningDataModule):
    def __init__(self, 
        ssl_task: SSLTask = RelativePositioning,
        window_len_s=10, 
        random_state=9, 
        batch_size: int = 64, 
        num_workers=0,
        data_dir='data',
        cache_dir=None,
        datasets:list[str]=None,
        overwrite_preprocessed=False,
    ):
        super().__init__()
        self.ssl_task = ssl_task
        self.window_len_s = window_len_s
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.random_state = random_state
        self.data_dir = data_dir
        self.cache_dir = cache_dir if cache_dir is not None else data_dir
        self.overwrite_preprocessed = overwrite_preprocessed
        HBN_DSNUMBERS = ['ds005514','ds005512','ds005511','ds005510','ds005509','ds005508','ds005507','ds005506','ds005505']
        self.datasets = datasets if datasets is not None else HBN_DSNUMBERS
        self.save_hyperparameters()

    def prepare_data(self):
        # create preprocessed data if not exists
        print(f"Using datasets: {self.datasets}")
        selected_tasks = ['RestingState']
        for dsnumber in self.datasets:
            savedir = f'{self.cache_dir}/{dsnumber}_preprocessed'
            if not os.path.exists(savedir) or self.overwrite_preprocessed:
                ds = HBNDataset(dsnumber, data_path=f"{self.data_dir}/{dsnumber}", tasks=selected_tasks, num_workers=-1, preload=False)
                ds = self.preprocess(ds, savedir)
        
    def preprocess(self, ds, savedir):
        from sklearn.preprocessing import scale as standard_scale
        os.makedirs(savedir, exist_ok=True)

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
        preprocess(ds, preprocessors, save_dir=savedir, overwrite=True, n_jobs=-1)

        return ds

    def setup(self, stage=None):
        all_ds = BaseConcatDataset([load_concat_dataset(path=f'{self.cache_dir}/{dsnumber}_preprocessed', preload=False) for dsnumber in self.datasets])
        # set desired label target
        target_name = 'age'
        for ds in all_ds.datasets:
            ds.target_name = target_name

        # Extract windows
        fs = all_ds.datasets[0].raw.info['sfreq']
        window_len_samples = int(fs * self.window_len_s)
        window_stride_samples = int(fs * self.window_len_s) # non-overlapping
        self.windows_ds = create_fixed_length_windows(
            all_ds, start_offset_samples=0, stop_offset_samples=None,
            window_size_samples=window_len_samples,
            window_stride_samples=window_stride_samples, drop_last_window=True,
            preload=False)

        # split into train/valid/test by subjects
        # Note: right now ignore train split. Train on all subjects
        subjects = np.unique(self.windows_ds.description['subject'])
        subj_train, subj_test = train_test_split(
            subjects, test_size=0.4, random_state=self.random_state)
        subj_valid, subj_test = train_test_split(
            subj_test, test_size=0.5, random_state=self.random_state)
        self.split_ids = {'train': subj_train, 'valid': subj_valid, 'test': subj_test}

        if stage == 'fit':
            # use all datasets for training
            self.train_ds = self.ssl_task.dataset(self.windows_ds.datasets)
            self.valid_ds = self.ssl_task.dataset(
                [ds for ds in self.windows_ds.datasets
                if ds.description['subject'] in self.split_ids['valid']])
            self.valid_ds.return_pair = False
        elif stage == 'test':
            self.test_ds = self.ssl_task.dataset(
                [ds for ds in self.windows_ds.datasets
                if ds.description['subject'] in self.split_ids['test']])
            self.test_ds.return_pair = False

    def train_dataloader(self):
        train_sampler = self.ssl_task.sampler(self.train_ds)
        if not dist.is_initialized():
            print(f"Number of datasets: {len(self.train_ds.datasetes)}")
            print(f"Number of examples: {train_sampler.n_examples}")
        return DataLoader(self.train_ds, sampler=train_sampler, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        pass

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        pass

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer 

class HBNDataset(BaseConcatDataset):
    """A class for Health Brain Network datasets.

    Parameters
    ----------
    dataset_name: str
        name of dataset included in eegdash to be fetched
    subject_ids: list(int) | int | None
        (list of) int of subject(s) to be fetched. If None, data of all
        subjects is fetched.
    dataset_kwargs: dict, optional
        optional dictionary containing keyword arguments
        to pass to the moabb dataset when instantiating it.
    dataset_load_kwargs: dict, optional
        optional dictionary containing keyword arguments
        to pass to the moabb dataset's load_data method.
        Allows using the moabb cache_config=None and
        process_pipeline=None.
    """

    def __init__(
        self,
        dataset_name: str,                                  # dataset name (dsnumber in BIDS)
        data_path: str = '/mnt/nemar/openneuro',            # path to dataset
        subjects: list[int] | int | None = None,            # subject ids to fetch. Default None fetches all
        tasks: list[int] | int | None = None,
        preload: bool = False,
        num_workers: int = 1,
        dataset_kwargs: dict[str, Any] | None = None,
        dataset_load_kwargs: dict[str, Any] | None = None,
    ):
        self.bids_dataset = BIDSDataset(data_dir=data_path, dataset=dataset_name)
        subject_df = self.bids_dataset.subjects_metadata
        def parseBIDSfile(f):
            raw = mne.io.read_raw_eeglab(f, preload=preload)
            metadata_keys = ['task', 'session', 'run', 'subject', 'sfreq']
            metadata = {key: getattr(self.bids_dataset, key)(f) for key in metadata_keys}
            subject = self.bids_dataset.subject(f)
            subject_metadata_keys = ['age', 'sex', 'ehq_total', 'p_factor', 'attention', 'internalizing', 'externalizing']
            metadata.update({key: subject_df.loc[subject_df['participant_id'] == f"sub-{subject}"][key].values[0] for key in subject_metadata_keys})
            # # electrodes locations in 2D
            # lt = mne.channels.find_layout(raw.info, 'eeg')
            # x, y = lt.pos[:,0], lt.pos[:,1]
            # metadata['electrodes_xy'] = np.array([x, y]).T
            return BaseDataset(raw, metadata)

        files = self.bids_dataset.get_files()
        # filter files
        if subjects:
            if type(subjects) == int:
                all_subjects = self.bids_dataset.subjects
                subjects = all_subjects[:subjects]
                files = [f for f in files if any(subject in f for subject in subjects)]
            else:
                files = [f for f in files if any(subject in f for subject in subjects)]
        if tasks:
            files = [f for f in files if any(task in f for task in tasks)]

        # parallel vs serial execution
        if num_workers == 1:
            all_base_ds = []
            for f in files:
                base_ds = parseBIDSfile(f)
                if base_ds:
                    all_base_ds.append(base_ds)
        else:
            all_base_ds = Parallel(n_jobs=num_workers)(
                    delayed(parseBIDSfile)(f) for f in files
            )
        super().__init__(all_base_ds)
    
    def load_data(self, fname):
        from mne.io.eeglab._eeglab import _check_for_scipy_mat_struct
        from scipy.io import loadmat
        eeglab_fields = ['setname','filename','filepath','subject','group','condition','session','comments','nbchan','trials','pnts','srate','xmin','xmax','times','icaact','icawinv','icasphere','icaweights','icachansind','chanlocs','urchanlocs','chaninfo','ref','event','urevent','eventdescription','epoch','epochdescription','reject','stats','specdata','specicaact','splinefile','icasplinefile','dipfit','history','saved','etc']
        eeg = loadmat(fname, squeeze_me=True, mat_dtype=False, variable_names=eeglab_fields)
        eeg['data'] = fname
        return _check_for_scipy_mat_struct(eeg)
        
class BIDSDataset():
    ALLOWED_FILE_FORMAT = ['eeglab', 'brainvision', 'biosemi', 'european']
    RAW_EXTENSION = {
        'eeglab': '.set',
        'brainvision': '.vhdr',
        'biosemi': '.bdf',
        'european': '.edf'
    }
    METADATA_FILE_EXTENSIONS = ['eeg.json', 'channels.tsv', 'electrodes.tsv', 'events.tsv', 'events.json']
    def __init__(self,
            data_dir=None,                            # dataset directory
            dataset='',                               # dataset name (e.g. ds005505)
            raw_format='eeglab',                      # format of raw data
        ):                            
        if data_dir is None or not os.path.exists(data_dir):
            raise ValueError('data_dir must be specified and must exist')
        self.bidsdir = Path(data_dir)
        self.dataset = dataset

        if raw_format.lower() not in self.ALLOWED_FILE_FORMAT:
            raise ValueError('raw_format must be one of {}'.format(self.ALLOWED_FILE_FORMAT))
        self.raw_format = raw_format.lower()

        # get all .set files in the bids directory
        temp_dir = (Path().resolve() / 'data')
        if not os.path.exists(temp_dir):
            os.mkdir(temp_dir)
        if not os.path.exists(temp_dir / f'{dataset}_files.npy'):
            self.files = self.get_files_with_extension_parallel(self.bidsdir, extension=self.RAW_EXTENSION[self.raw_format])
            np.save(temp_dir / f'{dataset}_files.npy', self.files)
        else:
            self.files = np.load(temp_dir / f'{dataset}_files.npy', allow_pickle=True)

    @property
    def subjects_metadata(self):
        subject_file = self.bidsdir / 'participants.tsv'
        if not os.path.exists(subject_file):
            raise ValueError('participants.tsv file not found in dataset')
        else:
            subjects = pd.read_csv(subject_file, sep='\t')
            return subjects
    
    @property
    def subjects(self):
        return self.subjects_metadata['participant_id'].values

    def get_property_from_filename(self, property, filename):
        import platform
        if platform.system() == "Windows":
            lookup = re.search(rf'{property}-(.*?)[_\\]', filename)
        else:
            lookup = re.search(rf'{property}-(.*?)[_\/]', filename)
        return lookup.group(1) if lookup else ''

    def get_bids_file_inheritance(self, path, basename, extension):
        '''
        Get all files with given extension that applies to the basename file 
        following the BIDS inheritance principle in the order of lowest level first
        @param
            basename: bids file basename without _eeg.set extension for example
            extension: e.g. channels.tsv
        '''
        top_level_files = ['README', 'dataset_description.json', 'participants.tsv']
        bids_files = []

        # check if path is str object
        if isinstance(path, str):
            path = Path(path)
        if not path.exists:
            raise ValueError('path {path} does not exist')

        # check if file is in current path
        for file in os.listdir(path):
            # target_file = path / f"{cur_file_basename}_{extension}"
            if os.path.isfile(path/file):
                cur_file_basename = file[:file.rfind('_')]
                if file.endswith(extension) and cur_file_basename in basename:
                    filepath = path / file
                    bids_files.append(filepath)

        # check if file is in top level directory
        if any(file in os.listdir(path) for file in top_level_files):
            return bids_files
        else:
            # call get_bids_file_inheritance recursively with parent directory
            bids_files.extend(self.get_bids_file_inheritance(path.parent, basename, extension))
            return bids_files

    def get_bids_metadata_files(self, filepath, metadata_file_extension):
        """
        (Wrapper for self.get_bids_file_inheritance)
        Get all BIDS metadata files that are associated with the given filepath, following the BIDS inheritance principle.
        
        Args:
            filepath (str or Path): The filepath to get the associated metadata files for.
            metadata_files_extensions (list): A list of file extensions to search for metadata files.
        
        Returns:
            list: A list of filepaths for all the associated metadata files
        """
        if isinstance(filepath, str):
            filepath = Path(filepath)
        if not filepath.exists:
            raise ValueError('filepath {filepath} does not exist')
        path, filename = os.path.split(filepath)
        basename = filename[:filename.rfind('_')]
        # metadata files
        meta_files = self.get_bids_file_inheritance(path, basename, metadata_file_extension)
        if not meta_files:
            raise ValueError('No metadata files found for filepath {filepath} and extension {metadata_file_extension}')
        else:
            return meta_files
        
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

    def load_raw(self, raw_file, preload=False):
        print(f"Loading {raw_file}")
        if raw_file.endswith('.set'):
            EEG = mne.io.read_raw_eeglab(raw_file, preload=preload)
        else:
            EEG = mne.io.Raw(raw_file, preload=preload)
        return EEG
    
    def get_files(self):
        return self.files
    
    def resolve_bids_json(self, json_files: list):
        """
        Resolve the BIDS JSON files and return a dictionary of the resolved values.
        Args:
            json_files (list): A list of JSON files to resolve in order of leaf level first

        Returns:
            dict: A dictionary of the resolved values.
        """
        if len(json_files) == 0:
            raise ValueError('No JSON files provided')
        json_files.reverse() # TODO undeterministic

        json_dict = {}
        for json_file in json_files:
            with open(json_file) as f:
                json_dict.update(json.load(f))
        return json_dict

    def sfreq(self, data_filepath):
        json_files = self.get_bids_metadata_files(data_filepath, 'eeg.json')
        if len(json_files) == 0:
            raise ValueError('No eeg.json found')

        metadata = self.resolve_bids_json(json_files)
        if 'SamplingFrequency' not in metadata:
            raise ValueError('SamplingFrequency not found in metadata')
        else:
            return metadata['SamplingFrequency']
    
    def electrodes_xy(self, data_filepath, use_bids=False):
        if use_bids:
            try:
                electrodes_files = self.get_bids_metadata_files(data_filepath, 'electrodes.tsv')
                electrodes = pd.read_csv(electrodes_files[0], sep='\t')
                # TODO this is totally wrong. Just a placeholder to get the pipeline flow
                # interpret electrodes location using coordsystem and use correct projection function
                coordsystem = self.get_bids_metadata_files(data_filepath, 'coordsystem.json')
                x = np.asarray(electrodes['x'])
                y = np.asarray(electrodes['y'])
                z = np.asarray(electrodes['z'])
                # to create mne info and base raw
                # https://mne.tools/stable/generated/mne.create_info.html#mne.create_info
                # https://mne.tools/stable/generated/mne.io.BaseRaw.html#mne.io.BaseRaw
                from eeg_positions.utils import _stereographic_projection
                x, y = _stereographic_projection(x,y,z)
            except ValueError:
                warnings.warn('No electrodes.tsv found. Attempt to extract electrodes locations from raw file')
                EEG = self.load_raw(data_filepath)
                # get 2D electrodes locations from mne Layout
                # https://mne.tools/stable/auto_tutorials/intro/40_sensor_locations.html
                lt = mne.channels.find_layout(EEG.info, 'eeg')
                x, y = lt.pos[:,0], lt.pos[:,1]
        else:
            EEG = self.load_raw(data_filepath)
            EEG = EEG.pick('eeg')
            # get 2D electrodes locations from mne Layout
            # https://mne.tools/stable/auto_tutorials/intro/40_sensor_locations.html
            lt = mne.channels.find_layout(EEG.info)
            x, y = lt.pos[:,0], lt.pos[:,1]
        
        return x, y

    def task(self, data_filepath):
        return self.get_property_from_filename('task', data_filepath)
        
    def session(self, data_filepath):
        return self.get_property_from_filename('session', data_filepath)

    def run(self, data_filepath):
        return self.get_property_from_filename('run', data_filepath)

    def subject(self, data_filepath):
        return self.get_property_from_filename('sub', data_filepath)
    
if __name__ == "__main__":
    dataset = HBNDataset(
        dataset_name = "ds004186", # ds004186 ds005510
    )