import os

import numpy as np
import scipy.io
import mne
import torch
from tqdm import tqdm
import hashlib
from abc import ABC, abstractmethod
import pickle

class SSLTransform(ABC):
    def __init__(self, x_params):
        """
        @param dict x_params {
                sfreq: int 
                cache_dir: str
                win: int | self.SFREQ
                stride: int | self.SFREQ/2
                tau_pos: int | self.SFREQ*3
                tau_neg: int | self.SFREQ*3
                n_samples: int | 1
                seed: int | 0
                isResume: bool | True
            }
        """
        self.SFREQ = x_params['sfreq']
        default_params = {
            "cache_dir": '/expanse/projects/nemar/dtyoung/eeg-ssl/data/childmind-rest-cache',
            "win": self.SFREQ,
            "stride": self.SFREQ/2,
            "tau_pos": int(self.SFREQ*3),
            "tau_neg": int(self.SFREQ*3),
            "n_samples": 1,
            "seed": 0,
            "isResume": True
        }

        default_params.update(x_params)
        for k,v in default_params.items():
            setattr(self, k, v)

        # validity check of inputs
        if self.stride > self.tau_pos:
            raise ValueError('Stride should not be larger than positive window size')

    @property
    @abstractmethod
    def data_keys(self):
        pass

    @property
    @abstractmethod
    def method(self):
        pass

    def process_data(self, raw_file, key):
        samples, labels = self.retrieve_cache(key)
        if np.any(samples):
            return samples, labels
        else:
            data = self.preload_raw(raw_file)
            return self.transform(data, key)
        
    def preload_raw(self, raw_file):
        EEG = mne.io.read_raw_eeglab(os.path.join(raw_file), preload=True)
        mat_data = EEG.get_data()

        if len(mat_data.shape) > 2:
            raise ValueError('Expect raw data to be CxT dimension')
        return mat_data

    @abstractmethod
    def transform(self, data, key):
        """
        @param dict data
        @param str key   - unique key for caching
        @return np.array - time x features
        """
        pass
    
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


class RelativePositioning(SSLTransform):
    data_keys = ["feat_inst_theta", "feat_inst_alpha", "feat_inst_beta"] # R G B
    method = "RP"

    def __init__(self, x_params):
        super().__init__(x_params)

    def transform(self, data, key):
        '''
        For each n_samples, get the anchor window,
        then choose corresponding positive windows (ones whose onsets is less then tau_pos from anchor start)
        and corresponding negative windows (ones whose onsets is larger than tau_neg from anchor_start)
        @param data - multi-channel time series C x T
        @param key - ID associated with the time-series data sample

        @return
            samples: list of samples, each has dim 2 x C x W where it has an anchor and a positive/negative sample
            labels: list of labels associated with each sample
        '''
        samples = []
        labels = []

        for anchor_start in np.arange(0, data.shape[1]-self.win, self.win): # non-overlapping anchor window
            # Positive window start t_pos:
            #     - |t_pos - t_anchor| <= tau_pos
            #           <-> t_pos <= tau_pos + t_anchor
            #           <-> t_pos => t_anchor - tau_pos
            #     - t_pos < T - win
            #.    - t_pos > 0            
            pos_winds_start = np.arange(np.maximum(0, anchor_start - self.tau_pos), np.minimum(anchor_start+self.tau_pos, data.shape[1]-self.win), self.win) # valid positive samples onsets
            if len(pos_winds_start) > 0:
                # positive context                
                pos_winds = [data[:, sample_start:sample_start+self.win] for sample_start in np.random.choice(pos_winds_start, self.n_samples, replace=False)]
                anchors = [data[:,anchor_start:anchor_start+self.win] for i in range(len(pos_winds))] # repeat same anchor window
                samples.extend([np.array([anchors[i], pos_winds[i]]) for i in range(len(anchors))]) # if anchors[i].shape == pos_winds[i].shape])
                labels.extend(np.ones(len(anchors)))

                # negative context
                # Negative window start t_neg:
                #     - |t_neg - t_anchor| > tau_neg
                #           <-> t_neg > tau_neg + t_anchor
                #           <-> t_neg < t_anchor - tau_neg
                #     - t_neg < T - win
                #.    - t_neg > 0
                neg_winds_start = np.concatenate((np.arange(0, anchor_start-self.tau_neg, self.win), np.arange(anchor_start+self.tau_neg, data.shape[1]-self.win, self.win)))
                neg_winds = np.array([data[:,sample_start:sample_start+self.win] for sample_start in np.random.choice(neg_winds_start, self.n_samples, replace=False)])
                samples.extend([np.array([anchors[i], neg_winds[i]]) for i in range(len(anchors))]) # if anchors[i].shape == neg_winds[i].shape])
                labels.extend(np.zeros(len(anchors)))

        samples = np.stack(samples) # N x 2 (anchors, pos/neg) x C x W
        if len(samples) != len(labels):
            raise ValueError('Number of samples and labels mismatch')

        self.cache_data(samples, labels, key)
        return samples, np.array(labels)

class TemporalShuffling(SSLTransform):
    data_keys = ["feat_inst_theta", "feat_inst_alpha", "feat_inst_beta"] # R G B
    method = 'TS'

    def __init__(self, x_params):
        super().__init__(x_params)

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

class CPC(SSLTransform):
    data_keys = ["feat_inst_theta", "feat_inst_alpha", "feat_inst_beta"] # R G B
    method = 'CPC'

    def __init__(self, x_params):
        super().__init__(x_params)

    def transform(self, data, key):
        '''
        For all steps, raw EEG window is passed through an encoder to get latent representation
        For each series of Nc contiguously non-overlapping context windows, an autoregressive model is applied to get a context embedding
        An array of Np contigously non-overlapping windows that follows the context group is used as future (positive) samples
        For each positive sample, pick a list of Nb negative samples randomly from the data
        '''
        # Adopting the (Banville et al, 2020) practice, ...
        # for each subject dataset, get all possible context and future samples pairs. --> Nb+1 pairs
        # then for each Np future sample, pick negative samples from the other Nb pairs
        # => Nb is determined by Nc, Np and window size
        context_windows = []
        future_windows = []
        for ti in arange(0, data.shape[1]-self.win*(self.Nc+self.Np), self.win*self.Nc):
            # ti is the index of the first window in the context array
            context_windows.append(data[:, np.arange(ti, ti+(self.win*self.Nc), self.win)])
            future_windows.append(data[:, np.arange(ti+(self.win*self.Nc), i+(self.win*(self.Nc+self.Np)), self.win)])
        
        negative_windows = []
        for i in range(len(future_windows)):
            negative_windows.append([np.random.choice(arr, replace=False) for arr in future_windows[:i] + future_windows[i+1:]])
        
        return list(zip(context_windows, future_windows, negative_windows))

class ChildmindSSLDataset(torch.utils.data.Dataset):
    SFREQ = 128

    def __init__(self,
            data_dir='/expanse/projects/nemar/dtyoung/eeg-ssl/data/childmind-rest', # location of asr cleaned data 
            subjects:list=None,                                       # subjects to use, default to all
            n_subjects=None,                                          # number of subjects to pick, default all
            x_params={
                "feature": "RelativePositioning",                    #
                "window": -1,                                         # number of samples to average over (-1: full subject)
                "stride": 1,                                          # number of samples to stride window by (does nothing when window = -1)
            },
            n_cv=None,                                                # (k,folds) Use this to control train vs test; independent of seed
            is_test=False,                                            # use (folds-1 or 1 fold) if n_cv != None
            seed=None):                                               # numpy random seed
        np.random.seed(seed)
        self.basedir = data_dir
        self.files = np.array([i for i in os.listdir(self.basedir) if i.split('.')[-1] == 'set'])
        self.subjects = np.array([i.split('_')[0] for i in os.listdir(self.basedir) if i.split('.')[-1] == 'set'])
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

        # Split Train-Test
        self.n_cv = n_cv
        self.is_test = is_test
        if self.n_cv is not None:
            split_size = int(n_subjects / self.n_cv[1])
            if not self.is_test:
                self.subjects = self.subjects[:self.n_cv[0]*split_size] + \
                                self.subjects[(self.n_cv[0]+1)*split_size:]
            else:
                self.subjects = self.subjects[self.n_cv[0]*split_size:(self.n_cv[0]+1)*split_size]

        # Instantiate transformer
        if "sfreq" not in x_params:
            x_params['sfreq'] = self.SFREQ
        self.x_params = x_params
        if type(x_params["feature"]) is str:
            try:
                transformer_cls = globals()[x_params["feature"]]
            except KeyError:
                raise ValueError("x_params.feature class not found")
        else:
            transformer_cls = x_params["feature"]
        self.x_transformer = transformer_cls(self.x_params)

        # Process data
        data_labels = [self.__transform_raw(i) for i in tqdm(self.files)]
        self.data = [i[0] for i in data_labels]
        self.y_data = [i[1] for i in data_labels]
        self.__aggregate_data()
        self.__shuffle_data()

    def __transform_raw(self, raw_file):
        # Transform data (populates self.data and self.ch_names). Note: this modifies input data.
        return self.x_transformer.process_data(os.path.join(self.basedir, raw_file), key=raw_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        # data = torch.stack(data)
        return data, self.y_data[idx]

    def __aggregate_data(self):
        '''
        Assume that self.data is currently subject-centric, not sample-centric.
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
        self.y_data = np.concatenate(self.y_data, axis=0)

        if len(self.data) != len(self.y_data):
            raise ValueError('Unmatched labels-samples')
        
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
        self.y_data = self.y_data[shuffle_idxs]
        self.subjects = self.subjects[shuffle_idxs]
        self.files = self.files[shuffle_idxs]

        if len(self.data) != len(self.y_data):
            raise Exception('Mismatch between number of samples and labels')
