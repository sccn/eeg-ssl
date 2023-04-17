import os

import mne
import numpy as np
import scipy
from sklearn.preprocessing import MinMaxScaler
import torch
import torchvision.models as torchmodels
from tqdm import tqdm
from matplotlib import pyplot as plt

from abc import ABC, abstractmethod

class BIDSTransform(ABC):
    def __init__(self, x_params):
        self.SFREQ = 256

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

class SSLTransform(BIDSTransform):

    def __init__(self, x_params):
        """
        @param dict x_params {
                method: str | "TS"
                window: int | self.SFREQ
                stride: int | self.SFREQ/2
                tau_pos: int | self.SFREQ*3
                tau_neg: int | self.SFREQ*3
                n_samples: int | 2
            }
        """
        super().__init__(x_params)
        self.method = x_params["method"] if "method" in x_params else "TS" # default to temporal shuffling
        self.window = self.SFREQ if x_params["window"] < 1 else x_params["window"]
        self.stride = self.SFREQ/2 if self.window < 1 else x_params["stride"]
        self.tau_pos = x_params["tau_pos"] if "tau_pos" in x_params else self.window*3 # arbitrary default
        self.tau_neg = x_params["tau_neg"] if "tau_neg" in x_params else self.window*3 # arbitray default
        self.n_samples = x_params["n_samples"] if "n_samples" in x_params else 2 # arbitrary default
        self.seed = x_params["seed"]
        np.random.seed(self.seed)

    def relative_positioning(self, data):
        '''
        For each n_samples, get the anchor window,
        then choose corresponding positive windows (ones whose onsets is less then tau_pos from anchor start)
        and corresponding negative windows (ones whose onsets is larger than tau_neg from anchor_start)
        @parameters
            data: F x T x Ch
        '''
        samples = []
        labels = []
        data = np.array([data[k] for k in self.data_keys]) # stack all frequency bands

        for anchor_start in np.arange(0, data.shape[1], self.window): # non-overlapping anchor window
            pos_winds_start = np.arange(anchor_start, np.minimum(anchor_start+self.tau_pos, data.shape[1]-self.window), self.stride) # valid positive samples onsets
            if len(pos_winds_start) > 0:
                pos_winds = [data[:, sample_start:sample_start+self.window,:] for sample_start in np.random.choice(pos_winds_start, self.n_samples, replace=False)]
                anchors = [data[:,anchor_start:anchor_start+self.window,:] for i in range(len(pos_winds))] # repeat same anchor window
                samples.extend([np.array([anchors[i], pos_winds[i]]) for i in range(len(anchors))]) # if anchors[i].shape == pos_winds[i].shape])
                labels.extend(np.ones(len(anchors)))

                # for negative windows, want both sides of anchor window
                neg_winds_start = np.concatenate((np.arange(0, anchor_start-self.tau_neg-self.window, self.stride), np.arange(anchor_start+self.tau_neg, data.shape[1]-self.window, self.stride)))
                neg_winds = np.array([data[:,sample_start:sample_start+self.window,:] for sample_start in np.random.choice(neg_winds_start, self.n_samples, replace=False)])
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
        data = np.array([data[k] for k in self.data_keys]) # stack all frequency bands

        tau_pos = self.tau_pos
        for pos_start in np.arange(0, data.shape[1], tau_pos): # non-overlapping positive contexts
            pos_winds = [data[:, pos_start:pos_start+self.window,:], data[:, pos_start+self.window*2:pos_start+self.window*3,:]] # two positive windows,
            inorder = np.array(pos_winds[:1] + [data[:, pos_start+self.window:pos_start+self.window*2,:]] + pos_winds[1:])
            samples.extend([inorder, np.flip(inorder).copy()])
            labels.extend(np.ones(2))

            # for negative windows, want both sides of anchor window
            neg_winds_start = np.concatenate((np.arange(0, pos_start-self.tau_neg-self.window, self.stride), np.arange(pos_start+tau_pos+self.tau_neg, data.shape[1]-self.window, self.stride)))
            selected_neg_start = np.random.choice(neg_winds_start, 1, replace=False)[0]
            disorder = np.array(pos_winds[:1] + [data[:,selected_neg_start:selected_neg_start+self.window,:]] + pos_winds[1:]) # two positive windows, disorder sample added to the end
            samples.extend([disorder, np.flip(disorder).copy()])
            labels.extend(np.zeros(2))

        samples = np.stack(samples)
        if len(samples) != len(labels):
            raise ValueError('Number of samples and labels mismatch')

        return samples, np.array(labels)

    def transform(self, data):
        # data is passed in as element in session array
        # data: K x T x C
        if self.method == "RP":
            data, labels = self.relative_positioning(data)
        if self.method == "TS":
            data, labels = self.temporal_shuffling(data)

        # data: S x P x K x T x C
        # map back to topo
        samples = []
        for i in range(data.shape[0]):
            tup = []
            for p in range(data.shape[1]):
                s = {}
                for k in range(data.shape[2]):
                    s[self.data_keys[k]] = data[i, p, k, :, :]
                feat_tensor = torch.tensor(self.generate_network_feat(s),
                    dtype=torch.float,
                    device="cuda" if torch.cuda.is_available() else "cpu") # T F
                feat_tensor = feat_tensor.transpose(1,3)
                feat_tensor = torch.nn.functional.interpolate(feat_tensor, size=(224,224))
                feat_tensor = torch.squeeze(feat_tensor)
                tup.append(feat_tensor)
            samples.append(tup)

        data = samples
        return data, labels

class SSLDataset(torch.utils.data.Dataset):
    SFREQ = 256

    def __init__(self,
            data_dir='/expanse/projects/nemar/dtyoung/eeg-ssl/data/ds004362_raw', # location of asr cleaned data bundled with markers and channels
            sessions:list=None,                                       # sessions to use, default to all
            n_sessions=None,                                          # number of sessions to pick, default all
            y_mode="bimodal", # {ordinal, split, bimodal}             # y to return (filtered on only first y_key) - ordinal: full range, split: <5 & >5, bimodal: <=3, >=7
            y_keys=["feltVlnc"],                                      # feltVlnc, feltArsl, feltCtrl, feltPred
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
        # only keep relevant keys
        mat_data = {k: v for k,v in mat_data.items() if (hasattr(self, "y_keys") and k in self.y_keys) or k == "ch_names" or k in self.x_transformer.data_keys}
        return mat_data

    def __transform_raw(self, data):
        # Transform channels to be consistent across sessions
        self.ch_names = self.x_transformer.info['ch_names']
        for n_session in range(len(data)):
            session_ch_names = [ch.strip() for ch in data[n_session]['ch_names']] # matlab forces fixed len str
            ch_order = [session_ch_names.index(i) for i in self.ch_names]
            for k in self.x_transformer.data_keys:
                data[n_session][k] = data[n_session][k][:, ch_order] # T Ch

        # Transform to feature space
        transformed = [self.x_transformer.transform(d) for d in tqdm(data)] # S T F..

        self.data = transformed[0][0]
        self.y_data = transformed[0][1]
        for d in range(1, len(transformed)):
            self.data   = np.concatenate([self.data, transformed[d][0]])
            self.y_data = np.concatenate([self.y_data, transformed[d][1]])

        # shuffle data
        p = np.random.permutation(len(self.data))
        self.data = [self.data[i] for i in p]
        self.y_data = [self.y_data[i] for i in p]

        if len(self.data) != len(self.y_data):
            raise Exception('Mismatch between number of samples and labels')


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.y_data[idx]