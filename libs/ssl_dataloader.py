import os

import numpy as np
import scipy.io
import torch
from tqdm import tqdm

from abc import ABC, abstractmethod

class AbstractTransform(ABC):
    def __init__(self, x_params):
        self.SFREQ = x_params['sfreq']

    @property
    @abstractmethod
    def data_keys(self):
        pass

    @abstractmethod
    def transform(self, data):
        """
        @param dict data
        @return np.array - time x features
        """
        pass

class SSLTransform(AbstractTransform):
    data_keys = ["feat_inst_theta", "feat_inst_alpha", "feat_inst_beta"] # R G B

    def __init__(self, x_params):
        """
        @param dict x_params {
                method: str | "TS"
                win: int | self.SFREQ
                stride: int | self.SFREQ/2
                tau_pos: int | self.SFREQ*3
                tau_neg: int | self.SFREQ*3
                n_samples: int | 2
            }
        """
        super().__init__(x_params)

        default_params = {
            "method": "RP",
            "cache_dir": "/expanse/projects/nemar/dtyoung/eeg-ssl/cache",
            "win": self.SFREQ,
            "stride": self.SFREQ/2,
            "tau_pos": self.SFREQ*3,
            "tau_neg": self.SFREQ*3,
            "n_samples": 1,
            "seed": 0
        }

        default_params.update(x_params)
        for k,v in default_params.items():
            setattr(self, k, v)

        # validity check of inputs
        if self.stride > self.tau_pos:
            raise ValueError('Stride should not be larger than positive window size')

        if self.method == "TS" and self.tau_pos < self.win*3:
            raise ValueError('Temporal shuffling requires positive context to be at least 3 times window size')

    def relative_positioning(self, data):
        '''
        For each n_samples, get the anchor window,
        then choose corresponding positive windows (ones whose onsets is less then tau_pos from anchor start)
        and corresponding negative windows (ones whose onsets is larger than tau_neg from anchor_start)
        '''
        samples = []
        labels = []

        for anchor_start in np.arange(0, data.shape[1], self.win): # non-overlapping anchor window
            pos_winds_start = np.arange(anchor_start+self.stride, np.minimum(anchor_start+self.tau_pos, data.shape[1]-self.win), self.stride) # valid positive samples onsets
            if len(pos_winds_start) > 0:
                pos_winds = [data[:, sample_start:sample_start+self.win] for sample_start in np.random.choice(pos_winds_start, self.n_samples, replace=False)]
                anchors = [data[:,anchor_start:anchor_start+self.win] for i in range(len(pos_winds))] # repeat same anchor window
                samples.extend([np.array([anchors[i], pos_winds[i]]) for i in range(len(anchors))]) # if anchors[i].shape == pos_winds[i].shape])
                labels.extend(np.ones(len(anchors)))

                # for negative windows, want both sides of anchor window
                neg_winds_start = np.concatenate((np.arange(0, anchor_start-self.tau_neg-self.win, self.stride), np.arange(anchor_start+self.tau_neg, data.shape[1]-self.win, self.stride)))
                neg_winds = np.array([data[:,sample_start:sample_start+self.win] for sample_start in np.random.choice(neg_winds_start, self.n_samples, replace=False)])
                samples.extend([np.array([anchors[i], neg_winds[i]]) for i in range(len(anchors))]) # if anchors[i].shape == neg_winds[i].shape])
                labels.extend(np.zeros(len(anchors)))

        samples = np.stack(samples)
        if len(samples) != len(labels):
            raise ValueError('Number of samples and labels mismatch')
        return samples, np.array(labels)

    def temporal_shuffling(self, data):
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

    def contrastive_predictive_coding(self, data):
        '''
        For all steps, raw EEG window is passed through an encoder to get latent representation
        For each series of Nc contiguously non-overlapping context windows, an autoregressive model is applied to get a context embedding
        An array of Np contigously non-overlapping windows that follows the context group is used as future (positive) samples
        For each positive sample, pick a list of Nb negative samples randomly from the data
        '''
        
        # Adopting the (Banville et al, 2020) practice, ...
        # for each session dataset, get all possible context and future samples pairs. --> Nb+1 pairs
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
            
    def transform(self, data, session):
        # data is passed in as element in session array
        # data: K x T x C
        if not os.path.exists(self.cache_dir):
            os.mkdir(self.cache_dir)
        if self.method == "CPC":
            if not os.path.exists(f'{self.cache_dir}/{session}_data.npy'):
                samples = self.contrastive_prective_coding(data)
                np.save(f'{self.cache_dir}/{session}_data', samples)
            else:
                samples = np.load(f'{self.cache_dir}/{session}_data.npy', allow_pickle=True) 
            return samples
        else
            if not (os.path.exists(f'{self.cache_dir}/{session}_data.npy') or os.path.exists(f'{self.cache_dir}/{session}_label.npy')):
                if self.method == "RP":
                    data, labels = self.relative_positioning(data)
                if self.method == "TS":
                    data, labels = self.temporal_shuffling(data)

                # data: S x W x K x T x C (nsample x nwindows/sample x channel x time)
                np.save(f'{self.cache_dir}/{session}_data', data)
                np.save(f'{self.cache_dir}/{session}_label', labels)
            else:
                data = np.load(f'{self.cache_dir}/{session}_data.npy', allow_pickle=True)
                labels = np.load(f'{self.cache_dir}/{session}_label.npy', allow_pickle=True)

            return data, labels

    def aggregate(self, transformed):
        '''
        Receive an array of SSL-transformed session data and aggregate them
        @parameters:
            transformed: array of SSL-transformed sessions data
        '''
        nchans = 62
        data = np.concatenate([transformed[i][0][:,:,0:nchans,:] for i in range(len(transformed))])
        y_data = np.concatenate([transformed[i][1] for i in range(len(transformed))])

        # shuffle data
        shuffle_idxs = np.random.permutation(len(data))
        data = data[shuffle_idxs]
        y_data = y_data[shuffle_idxs]

        if len(data) != len(y_data):
            raise Exception('Mismatch between number of samples and labels')

        return data, y_data


class BIDSDataset(torch.utils.data.Dataset):
    SFREQ = 160

    def __init__(self,
            data_dir='/expanse/projects/nemar/dtyoung/eeg-ssl/ds004362', # location of asr cleaned data 
            sessions:list=None,                                       # sessions to use, default to all
            n_sessions=None,                                          # number of sessions to pick, default all
            x_params={
                "feature": "SSLTransform",                    #
                "window": -1,                                         # number of samples to average over (-1: full session)
                "stride": 1,                                          # number of samples to stride window by (does nothing when window = -1)
            },
            balanced=False,                                           # within k-cv only; enforce class balance y_mode (only for split and bimodal); only on first y_key
            n_cv=None,                                                # (k,folds) Use this to control train vs test; independent of seed
            is_test=False,                                            # use (folds-1 or 1 fold) if n_cv != None
            seed=None):                                               # numpy random seed
        np.random.seed(seed)
        self.basedir = data_dir
        self.sessions = [i.split('.')[0] for i in os.listdir(self.basedir) if i.split('.')[-1] == 'mat']
        if sessions != None:
            self.sessions = [i for i in sessions if i in self.sessions]
            if len(sessions) - len(self.sessions) > 0:
                print("Warning: unknown keys present in user specified sessions")
        np.random.shuffle(self.sessions)
        n_sessions = n_sessions if n_sessions is not None else len(self.sessions)
        if n_sessions > len(self.sessions):
            print("Warning: n_sessions cannot be larger than sessions")
        self.sessions = self.sessions[:n_sessions]

        # Split Train-Test
        self.n_cv = n_cv
        self.is_test = is_test
        if self.n_cv is not None:
            split_size = int(n_sessions / self.n_cv[1])
            if not self.is_test:
                self.sessions = self.sessions[:self.n_cv[0]*split_size] + \
                                self.sessions[(self.n_cv[0]+1)*split_size:]
            else:
                self.sessions = self.sessions[self.n_cv[0]*split_size:(self.n_cv[0]+1)*split_size]

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

        # Preload data
        raw_data = [self.__preload_raw(session) for session in tqdm(self.sessions)]

        # Transform data (populates self.data and self.ch_names). Note: this modifies input data.
        self.__transform_raw(raw_data)


    def __preload_raw(self, session):
        # print(f"Preloading {session}...")
        mat_data = scipy.io.loadmat(os.path.join(self.basedir, session))
        mat_data = mat_data['data']
        return mat_data

    def __transform_raw(self, data):
        # Transform to feature space
        transformed = [self.x_transformer.transform(d, self.sessions[idx]) for idx, d in enumerate(tqdm(data))] # S T F..

        self.data, self.y_data = self.x_transformer.aggregate(transformed)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        # data = torch.stack(data)
        return data, self.y_data[idx]


