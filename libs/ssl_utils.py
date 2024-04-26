import torch
from torch.nn import functional as F
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from joblib import Parallel, delayed
import os
import mne

class MaskedContrastiveLearningDataset(torch.utils.data.Dataset):
    def __init__(self,
            data_dir='/mnt/nemar/dtyoung/eeg-ssl/data/childmind-rest', # location of asr cleaned data 
            subjects:list=None,                                       # subjects to use, default to all
            n_subjects=None,                                          # number of subjects to pick, default all
            x_params={
                "segment_length": 24,
                "sfreq": 128,
            },
            is_test=False,                                            # use (folds-1 or 1 fold) if n_cv != None
            seed=None):                                               # numpy random seed
        np.random.seed(seed)
        self.basedir = data_dir
        self.files = np.array([i for i in os.listdir(self.basedir) if i.split('.')[-1] == 'set'])
        self.subjects = np.array([i.split('_')[0] for i in os.listdir(self.basedir) if i.split('.')[-1] == 'set'])
        self.M = x_params['segment_length']
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


        # data_shape = None
        # label_shape = None
        # data = []
        # labels = []
        # for i in data_labels:
        #     if len(i[0]) > 0 and not data_shape:
        #         data_shape = i[0].shape
        #         label_shape = i[1].shape
        #     if len(i[0]) == 0:
        #         data.append(i[0].reshape(0,*data_shape[1:]))
        #         labels.append(i[1].reshape(0, *label_shape[1:]))
        #     else:
        #         data.append(i[0])
        #         labels.append(i[1])
        # self.data = data #[i[0] for i in data_labels]
        # self.y_data = labels #[i[1] for i in data_labels]
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

class LacunaSSL(torch.nn.Module):
    '''
    TODO
        Write model code
        Write loss function
        Write Dataloader
    '''
    def __init__(self, model_params):
        self.feature_encoder = self.build_feature_encoder()
        self.context_encoder = self.build_context_encoder()

    def build_feature_encoder(self):
        model = nn.Sequential(
            nn.Conv2D(3, 3, (4, 2)), # two rectangular 4 × 2 kernels
            nn.LeakyReLU, # each followed by a Leaky ReLU activation and Group Norm
            nn.GroupNorm
        )
        return model
    
    def build_context_encoder(self):
        input_size = 128
        hidden_size = 12
        model = nn.Sequential(
            nn.LSTM(input_size, hidden_size, bidirectional=True)
        )
        return model

class MaskedContrastiveLearningTask():
    def __init__(self, task_params=None):
        self.mask_size = task_params['mask_size']
        self.mask_probability = 0.5

    def forward(self, model, x):
        '''
        Forward pass of the model
        @parameter
            model:  nn.Module   model
            x    :  tensor      (N x K x C x T) batched segmented input
        @return
            prediction:         (N x D) Batch-size embeddings of the model's guess for masked inputs
            masked_latent:      (N x D) Batch-size embeddings of the feature encoder output of true masked inputs
            foil_latents:       (N x K x D) Batch-size embeddings of the feature conder output of the foil inputs
        '''
        embeddings = [model.feature_encoder(batch_segment) for batch_segment in x]
        embeddings = torch.stack(embeddings, dim=0) # K x N x F

        # learned masked vector embedding
        masked_vector_learned_embedding = torch.mean(embeddings, dim=0) # N x F

        # select from the sampled segment L masked inputs
        masked_indices = np.random.choice(embeddings.shape[0], size=(1, int(self.mask_probability*embeddings.shape[0])), replace=False)

        # replace the selected indices with the masked vector embedding
        true_masked_embeddings = embeddings[masked_indices].detach().clone() # L x N x F
        embeddings[masked_indices] = masked_vector_learned_embedding
        print('masked embeddings shape', embeddings.shape)

        # feed masked samples to context encoder. Every timestep has an output
        context_encoder_outputs = model.context_encoder(embeddings) # K x N x F
        print('context encoder outputs shape', context_encoder_outputs.shape)

        # context encoder_outputs of the masked input
        predicted_masked_latent = context_encoder_outputs[masked_indices] # L x N x F
        return predicted_masked_latent, true_masked_embeddings

    def loss(self, predictions, masked_latents):
        '''
        Follow implementation in https://github.com/dhruvbird/ml-notebooks/blob/main/nt-xent-loss/NT-Xent%20Loss.ipynb
        @parameter
            predictions:         (L x N x D) Batch-size embeddings of the model's guess for masked inputs
            masked_latents:      (L x N x D) Batch-size embeddings of the feature encoder output of masked inputs
        
        @return
            batched mean contrastive loss
        '''
        losses = []
        for i in range(masked_latents.shape[0]):
            predicted_masked_latent = predictions[i]
            masked_latent = masked_latents[i]
            foil_latents = masked_latents[torch.arange(masked_latents.shape[0]) != i]
            embbed_combined = torch.cat([torch.unsqueeze(masked_latent, dim=1), foil_latents[:,i]], dim=1).permute(0,2,1)
            print('combined shape permuted shape', embbed_combined.shape)
            embbed_combined = torch.cat([torch.unsqueeze(masked_latent, dim=1), foil_latents], dim=1).permute(0,2,1)
            # print('combined shape permuted shape', embbed_combined.shape) # N x D x K+1
            # print('masked latent', masked_latent[0,:])
            # print('equivalent first element of combined', embbed_combined[0,:,0])
            # print('is equivalent', embbed_combined[0,:,0] == masked_latent[0,:])
            cos_sim = F.cosine_similarity(torch.unsqueeze(predicted_masked_latent, dim=-1), embbed_combined, dim=1)
            # print('cosine similarity', cos_sim)
            labels = torch.zeros([cos_sim.shape[0], cos_sim.shape[-1]])
            labels[:,0] = 1
            # print('labels', labels)
            losses.append(F.cross_entropy(cos_sim, labels, reduction='mean'))
        # print('batch mean loss', loss)
        return torch.mean(losses)

    def train(self, model, dataset):
        num_epochs = self.train_params['num_epochs']
        batch_size = self.train_params['batch_size']
        optimizer  = self.train_params['optimizer']
        print_every = self.train_params['print_every']

        dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True)
        for e in range(num_epochs):
            for t, (samples, _) in enumerate(dataloader):
                predictions, masked_latents = self.forward(model, samples)
                loss = self.loss(predictions, masked_latents)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if t % print_every == 0:
                    # writer.add_scalar("Loss/train", loss.item(), e*len(dataloader)+t)
                    print('Epoch %d, Iteration %d, loss = %.4f' % (e, t, loss.item()))

                del predictions
                del masked_latents
                del loss

