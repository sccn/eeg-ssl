import os

import numpy as np
import scipy.io
import mne
import torch
from tqdm import tqdm
import hashlib
from abc import ABC, abstractmethod
import pickle
from joblib import Parallel, delayed

verbose = False
class MaskedContrastiveLearningDataset(torch.utils.data.Dataset):
    def __init__(self,
            data_dir='/mnt/nemar/dtyoung/eeg-ssl/data/childmind-rest', # location of asr cleaned data 
            subjects:list=None,                                       # subjects to use, default to all
            n_subjects=None,                                          # number of subjects to pick, default all
            x_params={
                "window": 24,
                "sfreq": 128,
            },
            is_test=False,                                            # use (folds-1 or 1 fold) if n_cv != None
            seed=None):                                               # numpy random seed
        np.random.seed(seed)
        self.basedir = data_dir
        self.files = np.array([i for i in os.listdir(self.basedir) if i.split('.')[-1] == 'set'])
        self.subjects = np.array([i.split('_')[0] for i in os.listdir(self.basedir) if i.split('.')[-1] == 'set'])
        self.M = x_params['window']
        if subjects != None:
            subjects = [i for i in subjects if i in self.subjects]
            selected_indices = [subjects.index(i) for i in subjects if i in self.subjects]
            self.subjects = np.array(subjects)[selected_indices]
            self.files = self.files[selected_indices]
            if len(subjects) - len(self.subjects) > 0:
                print("Warning: unknown keys present in user specified subjects")
        shuffling_indices = list(range(len(self.subjects)))
        np.random.shuffle(shuffling_indices)
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

        # Load raw data
        self.data = [self.transform_raw(i) for i in tqdm(self.files)] # here self.data has dimension of (n_files, C, T)

        # Process data

        # data_labels = [self.transform_raw(i) for i in tqdm(self.files)]
        self.data = Parallel(n_jobs=-1, backend="threading", verbose=1)(delayed(self._thread_worker)(i) for i in tqdm(self.files))
        # here self.data has dimension of (n_files, C, T)
        print('Data shape:', self.data[0].shape)

        self.__aggregate_data()
        self.__shuffle_data()

    '''
    Load and preprocess raw data
    @param str raw_file - path to raw file
    @return np.array - channel x time
    '''
    def transform_raw(self, raw_file):
        # Transform data (populates self.data and self.ch_names). Note: this modifies input data.
        data = self.preprocess_data(os.path.join(self.basedir, raw_file), key=raw_file)
        data = self.segment_input(data)
        if verbose:
            print('Segmented data shape', data.shape)

        return data

    def segment_input(self, x):    
        '''
        Split input into segments of M time sample
        @parameter
            x: (C x T) Multichannel EEG input
        @return
            output  (K x C x M) Multichannel EEG input segmented into K segments of length M
        '''
        # sample from left to right, discarding leftovers
        indices = np.arange(0, x.shape[-1]-self.M, self.M)
        samples = [x[:,idx:idx+self.M] for idx in indices]
        samples = np.stack(samples, axis=0)

        return samples

    def __aggregate_data(self):
        '''
        Assume that self.data is currently file-centric, not sample-centric.
        Flattening out self.data keeping sample-metadata association
        Currently tracked metadata:
            - Subject - self.subjects
            - Filename - self.files
            - Labels - self.y_data
        '''
        # best method shown by https://realpython.com/python-flatten-list/
        expanded_subjects = []
        for i, subj in enumerate(self.subjects):
            expanded_subjects.extend([subj]*len(self.data[i]))
        self.subjects = np.array(expanded_subjects)

        expanded_files = []
        for i, f in enumerate(self.files):
            expanded_files.extend([f]*len(self.data[i]))
        self.files = np.array(expanded_files)

        self.data = np.concatenate(self.data, axis=0)
        # self.y_data = np.concatenate(self.y_data, axis=0)

        # if len(self.data) != len(self.y_data):
        #     raise ValueError('Unmatched labels-samples')
        
        if len(self.files) != len(self.subjects) and len(self.files) != len(self.data):
            raise ValueError('Mismatch data-metadata')


    def __shuffle_data(self):
        '''
        Shuffle data preserving sample-metadata association
        Currently tracked metadata:
            - Subject - self.subjects
            - Filename - self.files
            - Labels - self.y_data
        '''
        # shuffle data
        shuffle_idxs = np.random.permutation(len(self.data))
        self.data = self.data[shuffle_idxs]
        # self.y_data = self.y_data[shuffle_idxs]
        self.subjects = self.subjects[shuffle_idxs]
        self.files = self.files[shuffle_idxs]

        # if len(self.data) != len(self.y_data):
        #     raise Exception('Mismatch between number of samples and labels')

    def _thread_worker(self, raw_file):
        return self.transform_raw(raw_file)

    def preprocess_data(self, raw_file, key):
        data = self.preload_raw(raw_file)
        # try:
        #     samples, labels = self.retrieve_cache(key)
        #     if np.any(samples):
        #         return samples, labels
        #     else:
        #         data = self.preload_raw(raw_file)
        #         return data
        # except:
        #     return np.array([]), np.array([])
        return data
        
    def preload_raw(self, raw_file):
        EEG = mne.io.read_raw_eeglab(os.path.join(raw_file), preload=True)
        mat_data = EEG.get_data()

        if len(mat_data.shape) > 2:
            raise ValueError('Expect raw data to be CxT dimension')
        return mat_data

    def hash_key(self, key):
        hash_object = hashlib.sha256()
        hash_object.update(bytes(key, 'utf-8'))

        # Calculate the hash (digest)
        hash_result = hash_object.digest()

        return hash_result.hex() # the hash result (in hexadecimal format)
    
    def retrieve_cache(self, key):
        data_hash = self.hash_key(key)
        if self.isResume and os.path.exists(f'{self.cache_dir}/{self.method}_{data_hash}.pkl'):
            print('Retrieve cache')
            cache_file = f'{self.cache_dir}/{self.method}_{data_hash}.pkl'
            with open(cache_file, 'rb') as fin:
                saved_data = pickle.load(fin) 
            samples = saved_data['data']
            labels = saved_data['labels']
            return samples, labels
        else:
            return np.empty(0), np.empty(0)

    def cache_data(self, data, labels, key):
        data_hash = self.hash_key(key)
        if not os.path.exists(self.cache_dir):
            os.mkdir(self.cache_dir)
        cache_file = f'{self.cache_dir}/{self.method}_{data_hash}.pkl' 
        with open(cache_file, 'wb') as fout:
            pickle.dump({'data': data, 'labels': labels}, fout)

        def default_return(msg=""):
            return msg

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        # return data, self.y_data[idx]
        return data


        if self.tau_pos < self.win*3:
            raise ValueError('Temporal shuffling requires positive context to be at least 3 times window size')

    def transform(self, data, key):
        '''
        Pick two samples from positive context
        Pick another sample either from positive context in between previous two (--> y = 1 (in-order)), or from negative context (--> y = 0 (disordered))
        Sample size is increased by switching positive samples
        x order: (pos_sample_left, in/dis-order_sample, pos_sample_right)
        '''
        # Our assumption will be the positive context will always be 3*window size.
        # Thus we'll have 3 consecutive windows extracted from positive context (in-order group)
        # And we'll also randomly pick one window in the negative context to form a disordered group
        samples = []
        labels = []

        tau_pos = self.tau_pos
        for pos_start in np.arange(0, data.shape[1], tau_pos): # non-overlapping positive contexts
            if pos_start + tau_pos < data.shape[1]:
                pos_winds = [data[:, pos_start:pos_start+self.win], data[:, pos_start+self.win*2:pos_start+self.win*3]] # two positive windows
                inorder = np.array(pos_winds[:1] + [data[:, pos_start+self.win:pos_start+self.win*2]] + pos_winds[1:])
                samples.extend([inorder, np.flip(inorder).copy()])
                labels.extend(np.ones(2))

                # for negative windows, want both sides of anchor window
                neg_winds_start = np.concatenate((np.arange(0, pos_start-self.tau_neg-self.win, self.stride), np.arange(pos_start+tau_pos+self.tau_neg, data.shape[1]-self.win, self.stride)))
                selected_neg_start = np.random.choice(neg_winds_start, 1, replace=False)[0]
                disorder = np.array(pos_winds[:1] + [data[:,selected_neg_start:selected_neg_start+self.win]] + pos_winds[1:]) # two positive windows, disorder sample added to the end
                samples.extend([disorder, np.flip(disorder).copy()])
                labels.extend(np.zeros(2))

        samples = np.stack(samples)
        if len(samples) != len(labels):
            raise ValueError('Number of samples and labels mismatch')

        return samples, np.array(labels)
