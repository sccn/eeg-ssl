import torch
from torch.nn import functional as F
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, random_split
import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from braindecode.datasets import BaseConcatDataset
<<<<<<< HEAD
=======

class RelativePositioningDataset(BaseConcatDataset):
    """BaseConcatDataset with __getitem__ that expects 2 indices and a target.
    """

    def __init__(self, list_of_ds):
        super().__init__(list_of_ds)
        self.return_pair = True

    def __getitem__(self, index):
        if self.return_pair:
            ind1, ind2, y = index
            return (super().__getitem__(ind1)[0],
                    super().__getitem__(ind2)[0]), y
        else:
            return super().__getitem__(index)

    @property
    def return_pair(self):
        return self._return_pair

    @return_pair.setter
    def return_pair(self, value):
        self._return_pair = value
>>>>>>> 496838cbdfc1608c7ea57ee0cbc76f86cbc084ea

class RelativePositioningDataset(BaseConcatDataset):
    """BaseConcatDataset with __getitem__ that expects 2 indices and a target.
    """

    def __init__(self, list_of_ds):
        super().__init__(list_of_ds)
        self.return_pair = True

    def __getitem__(self, index):
        if self.return_pair:
            ind1, ind2, y = index
            return (super().__getitem__(ind1)[0],
                    super().__getitem__(ind2)[0]), y
        else:
            return super().__getitem__(index)

    @property
    def return_pair(self):
        return self._return_pair

    @return_pair.setter
    def return_pair(self, value):
        self._return_pair = value
        
class MaskedContrastiveLearningTask():
    def __init__(self,
                dataset: torch.utils.data.Dataset,
                task_params={
                    'mask_prob': 0.5
                },
                train_params={
                    'num_epochs': 100,
                    'batch_size': 10,
                    'print_every': 10,
                    'learning_rate': 0.001,
                },
                is_cv=True,                  # whether to perform train-test-split on dataset. If False, train and val on the same dataset
                is_iterable=False,           # whether it's an iterable dataset
                random_seed=9,
                debug=True,
                verbose=False,
        ):
        torch.manual_seed(random_seed)
        self.dataset = dataset
        self.seed = random_seed
        self.masked_vector_learned_embedding = None
        self.train_params = train_params
        self.mask_probability = task_params['mask_prob']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.debug  = debug
        self.verbose=verbose
        self.is_cv = is_cv
        self.is_iterable = is_iterable
        self.train_test_split()

    def train_test_split(self):
        if self.is_cv:
            generator = torch.Generator().manual_seed(self.seed)
            self.dataset_train, self.dataset_val = torch.utils.data.random_split(self.dataset, [0.7,0.3], generator=generator)
        else:
            self.dataset_train = self.dataset_val = self.dataset

    def shuffle_batch(self, x, batch_dim=0):
        '''
        Shuffle sample in a batch to increase randomization
        '''
        shuffling_indices = list(range(x.shape[batch_dim]))
        np.random.shuffle(shuffling_indices)
        shuffling_indices = torch.tensor(shuffling_indices)
        x = torch.index_select(x, batch_dim, shuffling_indices)
        return x

    def forward(self, model, x):
        '''
        Forward pass of the model
        @parameter
            model:  nn.Module   model
            x    :  tensor      (N x C x T) batched raw input
        @return
            prediction:         (N x D x K) Batch-size embeddings of the model's guess for masked inputs
            masked_latent:      (N x D x K) Batch-size embeddings of the feature encoder output of true masked inputs
            foil_latents:       (N x D x K) Batch-size embeddings of the feature conder output of the foil inputs
        '''
        embeddings = model.feature_encoder(x) # N x D x K
                                              # forward pass of feature encoder generate intermediary embeddings
        if self.verbose:
            print('feature encoder output shape', embeddings.shape)

        if self.verbose:
            print('mask embedding value', model.mask_emb)

        # select from the sampled segment L masked inputs
        masked_indices = np.random.choice(embeddings.shape[-1], size=(int(self.mask_probability*embeddings.shape[-1]),), replace=False)
        if self.verbose:
            print('masked indices shape', masked_indices.shape)

        # replace the selected indices with the masked vector embedding
        true_masked_embeddings = embeddings[:,:,masked_indices].detach().clone() # N x D x K 
        if self.verbose:
            print('true masked embeddings shape', true_masked_embeddings.shape)

        for i in range(len(masked_indices)):
            embeddings[:,:,i] = model.mask_emb

        # feed masked samples to context encoder. Every timestep has an output
        context_encoder_outputs = model.context_encoder(embeddings) # N x D x K
        if self.verbose:
            print('context encoder outputs shape', context_encoder_outputs.shape)

        # context encoder_outputs of the masked input
        predicted_masked_latent = context_encoder_outputs[:,:,masked_indices] # N x D x K
        if self.verbose:
            print('predicted context encoder outputs shape', predicted_masked_latent.shape)
        return predicted_masked_latent, true_masked_embeddings

    def loss(self, predictions, masked_latents):
        '''
        Follow implementation in https://github.com/dhruvbird/ml-notebooks/blob/main/nt-xent-loss/NT-Xent%20Loss.ipynb
        @parameter
            predictions:         (N x D x K) Batch-size embeddings of the model's guess for masked inputs
            masked_latents:      (N x D x K) Batch-size embeddings of the feature encoder output of masked inputs

        @return
            batched mean contrastive loss
        '''
        losses = torch.zeros((masked_latents.shape[-1],), device=self.device)
        # contrastive learning is computed one masked sample at a time
        for k in range(masked_latents.shape[-1]):
            predicted_masked_latent = predictions[:,:,k] # N x D
            if self.verbose:
                print('predicted masked latent shape', predicted_masked_latent.shape)
            cos_sim = F.cosine_similarity(torch.unsqueeze(predicted_masked_latent, dim=-1), masked_latents, dim=1) # N x K
            if self.verbose:
                print('cosine similarity shape', cos_sim.shape)
            labels = torch.zeros([cos_sim.shape[0], cos_sim.shape[1]], device=self.device) # N x K
            labels[:,k] = 1
            # print('labels', labels)
            # losses.append(F.cross_entropy(cos_sim, labels, reduction='mean'))
            losses[k] = F.cross_entropy(cos_sim, labels, reduction='mean')
        if self.verbose:
            print('losses', losses)
        # return torch.mean(torch.tensor(losses))
        return torch.mean(losses)

    def train(self, model, train_params={}):
        print('Training on ', self.device)
        self.train_params.update(train_params)
        print("With parameters:", self.train_params)
        num_epochs = self.train_params['num_epochs']
        batch_size = self.train_params['batch_size']
        print_every = self.train_params['print_every']
        learning_rate = self.train_params['learning_rate']

        optimizer  = torch.optim.Adam(model.parameters(), lr=learning_rate)
        dataloader_train = DataLoader(self.dataset_train, batch_size = batch_size, shuffle = not self.is_iterable)
        model.to(device=self.device)
        model.train()
        for e in range(num_epochs):
            for t, samples in enumerate(dataloader_train):
                samples = self.shuffle_batch(samples, batch_dim=0)
                samples = samples.to(device=self.device, dtype=torch.float32)
                predictions, masked_latents = self.forward(model, samples)
                loss = self.loss(predictions, masked_latents)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if t % print_every == 0:
                    # writer.add_scalar("Loss/train", loss.item(), e*len(dataloader)+t)
                    print('Epoch %d, Iteration %d, loss = %.4f' % (e, t, loss.item()))

                if wandb.run is not None:
                    metrics = {"train/train_loss": loss.item()}
                    wandb.log(metrics)

                del samples
                del predictions
                del masked_latents
                del loss

            eval_train_score, eval_test_score = self.finetune_eval_score(model)

            if t % print_every == 0:
                # writer.add_scalar("Loss/train", loss.item(), e*len(dataloader)+t)
                print('Epoch %d, Iteration %d, val/train = %.4f, val/test = %.4f' % (e, t, eval_train_score, eval_test_score))

            if wandb.run is not None:
                wandb.log({"val/train_score": eval_train_score,
                           "val/test_score": eval_test_score})
        if wandb.run is not None:
            wandb.finish()

    def finetune_eval_score(self, model):
        model.eval()
        generator = torch.Generator().manual_seed(42)
        val_train, val_test = random_split(self.dataset_val, [0.7, 0.3], generator=generator)
        val_train_dataloader = DataLoader(val_train, batch_size = len(val_train), shuffle = True)
        val_test_dataloader = DataLoader(val_test, batch_size = len(val_test), shuffle = True)

        samples, labels = next(iter(val_train_dataloader))
        samples = samples.to(device=self.device, dtype=torch.float32)
        predictions = model(samples)
        # print(predictions)
        embeddings = torch.mean(predictions, dim=-1) # TODO is averaging the best strategy here, for classification?
        # print(embeddings)
        clf = LinearDiscriminantAnalysis()
        clf.fit(embeddings.detach().cpu().numpy(), labels.detach().cpu().numpy())
        train_score = clf.score(embeddings.detach().cpu().numpy(), labels.detach().cpu().numpy())
        print('Eval train score:', train_score)

        samples_test, labels_test = next(iter(val_test_dataloader))
        samples_test = samples_test.to(device=self.device, dtype=torch.float32)
        predictions = model(samples_test)
        embeddings = torch.mean(predictions, dim=-1) # TODO is averaging the best strategy here, for classification?
        test_score = clf.score(embeddings.detach().cpu().numpy(), labels_test.detach().cpu().numpy())
        print('Eval test score:', test_score)
        return train_score, test_score

from abc import ABC, abstractmethod
class SSLTask(ABC):
    DEFAULT_TASK_PARAMS = {
        'seed': 0,
    }
    def __init__(self, dataset, model, task_params=DEFAULT_TASK_PARAMS):
        self.dataset = dataset
        self.model = model
        self.seed = task_params['seed']
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        
    
    @property
    @abstractmethod
    def task_module(self) -> nn.Module:
        """Task specific module(s) to be trained"""
        pass

    @abstractmethod
    def get_task_model_params(self):
        pass

    @abstractmethod
    def forward(self, model, samples):
        pass

    @abstractmethod
    def criterion(self, predictions, labels):
        pass

import time
class RelativePositioning(SSLTask):
    DEFAULT_TASK_PARAMS = {
        'sfreq': 128,
        'win': 2,
        'tau_pos': 20,
        'tau_neg': 30,
        'n_samples': 1,
        'seed': 0,
    }
    def __init__(self,
                dataset: torch.utils.data.IterableDataset,
                model: torch.nn.Module,
                task_params=DEFAULT_TASK_PARAMS,
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                verbose=False
        ):
        super().__init__(dataset, model)

        self.dataset = dataset
        self.model = model
        task_params_final = self.DEFAULT_TASK_PARAMS.copy()
        task_params_final.update(task_params)
        # self.loss = nn.CrossEntropyLoss()

        for name, value in task_params_final.items():
            setattr(self, name, value)

        self.device = device
        self.verbose=verbose

        # initialize RP-specific layers
        self._add_layers()
    
    def _add_layers(self):
        D = 200
        sample = next(self.dataset.__iter__())
        C, T = sample.shape
        window_nsample = int(self.sfreq * self.win)
        with torch.no_grad():
            fake_input = torch.randn(1, C, window_nsample)
            embed_dim = torch.flatten(self.model(fake_input)).shape[0]
            self.flattened_encoder_embed_dim = embed_dim
        self.linear_ff = nn.Linear(embed_dim, D)
        self.loss_linear = nn.Linear(D, 2)
    
    @property
    def task_module(self) -> nn.Module:
        return nn.Sequential(self.linear_ff, self.loss_linear)

    def gRP(self, embeddings):
        '''
        @param
            embeddings - batch x 2 x D

        @return
            differences - batch x D
        '''
        return torch.abs(embeddings[:,0] - embeddings[:,1])

    def get_task_model_params(self):
        return list(self.linear_ff.parameters()) + list(self.loss_linear.parameters())

    def criterion(self, differences, labels):
        linear_combination = self.loss_linear(differences)

        # Calculate the loss
        # loss = torch.log(1 + torch.exp(-labels * linear_combination))
        # return loss.mean()
        return F.cross_entropy(linear_combination, labels)

    def forward(self, model, x):
        '''
        Relative positioning task:
            - For each anchor window, sample n_samples positive (before and after tau_pos) and negative windows
            - Negative context is anywhere in the sample that is more than tan_neg sample away from anchor window
        @param
            x: (N x C x T) batched raw input
        '''
        self.linear_ff.to(self.device).train()
        self.loss_linear.to(self.device).train()
        if self.verbose:
            print('x shape:', x.shape)


        window_nsample = int(self.sfreq * self.win)
        tau_pos_nsample = self.sfreq * self.tau_pos
        tau_pos_nsample_half = int(tau_pos_nsample / 2)
        tau_neg_nsample = self.sfreq * self.tau_neg

        # Pre-allocate the samples tensor
        # total_samples = self.n_samples * 2 * self.n_samples * x.shape[0]  # Positive and negative samples
        # samples = torch.empty(total_samples, 2, x.shape[1], window_nsample)
        # labels = torch.empty(total_samples, 1)

        samples = torch.Tensor()
        labels = torch.Tensor()
        # select n_samples anchor window randomly from entire recording
        # idx = 0
        for anchor_start in np.random.choice(np.arange(0, x.shape[2]-window_nsample, window_nsample), self.n_samples, replace=False):
            tau_pos_start = max(anchor_start - tau_pos_nsample_half, 0)
            tau_pos_end = min(anchor_start + window_nsample + tau_pos_nsample_half, x.shape[2]-window_nsample)
            tau_pos_winds = np.arange(tau_pos_start, tau_pos_end, window_nsample) 
            # sample positive samples from the window starting from tau_pos_nsample_half before anchor window to anchor_start+window_nsample+tau_pos_nsample_half
            if len(tau_pos_winds) > 0:
                for pos_wind_start in np.random.choice(tau_pos_winds, self.n_samples, replace=False):
                    # print('pos_wind_start', pos_wind_start)
                    anch  = x[:, :,anchor_start:anchor_start+window_nsample]
                    pos_w = x[:, :,pos_wind_start:pos_wind_start+window_nsample]


                    if self.verbose:
                        print('anchor shape:', anch.shape)   
                        # eeg_utils.plot_raw_eeg(anch[0][:15, :].cpu().numpy(), 128)
                        print('positive shape:', pos_w.shape)   
                        # eeg_utils.plot_raw_eeg(pos_w[0].cpu().numpy(), 128)

                    # samples[idx:idx+x.shape[0], 0] = anch
                    # samples[idx:idx+x.shape[0], 1] = pos_w
                    # labels[idx:idx+x.shape[0]] = 1

                    # idx += x.shape[0]

                    samples = torch.concat([samples, torch.stack([anch, pos_w], dim=1)]) # N x 2 x C x W
                    labels = torch.concat([labels, torch.ones(x.shape[0])]) 

            # samples - n_samples*N x 2 x C x W

            tau_neg_before = np.arange(0, anchor_start - tau_neg_nsample - window_nsample, window_nsample)
            tau_neg_after = np.arange(anchor_start + window_nsample + tau_neg_nsample, x.shape[2]-window_nsample, window_nsample)
            if len(tau_neg_before) > 0 or len(tau_neg_after) > 0:
                for neg_wind_start in np.random.choice(np.concatenate([tau_neg_before, tau_neg_after]), self.n_samples, replace=False):
                    # print('neg_wind_start', neg_wind_start)
                    anch  = x[:, :,anchor_start:anchor_start+window_nsample]
                    neg_w = x[:, :,neg_wind_start:neg_wind_start+window_nsample]

                    if self.verbose:
                        print('anchor shape:', anch.shape)   
                        # eeg_utils.plot_raw_eeg(anch[0][:15, :].cpu().numpy(), 128)
                        print('negative shape:', neg_w.shape)   
                        # eeg_utils.plot_raw_eeg(neg_w[0].cpu().numpy(), 128)

                    # samples[idx:idx+x.shape[0], 0] = anch
                    # samples[idx:idx+x.shape[0], 1] = neg_w
                    # labels[idx:idx+x.shape[0]] = 0

                    # idx += x.shape[0]
                    samples = torch.concat([samples, torch.stack([anch, neg_w], dim=1)]) # N x 2 x C x W
                    labels = torch.concat([labels, torch.zeros(x.shape[0])])
            # samples - n_samples*N x 2 x C x W

            # --> samples - 2*n_samples*N x 2 x C x W

        # if idx != total_samples:
        #     print('idx', idx, 'total_samples', total_samples)
        #     raise ValueError('Number of samples mismatch')

        if self.verbose:
            print('sample shape', samples.shape) # n_samples*2*n_samples*N x 2 x C x W

        samples = samples.to(device=self.device, dtype=torch.float32)
        embeddings = torch.stack([self.linear_ff(torch.flatten(model(samples[:, 0]), start_dim=1)), self.linear_ff(torch.flatten(model(samples[:, 1]), start_dim=1))], dim=1) # batch x 2 x D

        differences = self.gRP(embeddings)

        if len(differences) != len(labels):
            raise ValueError('Number of samples and labels mismatch')
        if len(torch.unique(labels)) == 1:
            print('Warning: All samples are of the same type')
        if not torch.all(torch.isin(torch.unique(labels), torch.tensor([0, 1]))):
            raise ValueError('Labels must be 0 or 1')

        labels = labels.to(dtype=torch.long, device=self.device)
        
        # shuffle data
        shuffle_indices = torch.randperm(len(differences))
        differences = differences[shuffle_indices]
        labels = labels[shuffle_indices]
        loss = self.criterion(differences, labels)

        del differences, labels

        return loss

class TemporalShuffling(SSLTask):
    DEFAULT_TASK_PARAMS = {
        'win': 50,
        'tau_pos': 150,
        'tau_neg': 170,
        'n_samples': 1,
    }
    def __init__(self,
                dataset: torch.utils.data.Dataset,
                model: torch.nn.Module,
                stride = 1,
                task_params=DEFAULT_TASK_PARAMS,
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                verbose=False
        ):
        super().__init__(dataset, model)
        self.dataset = dataset
        self.model = model

        task_params_final = self.DEFAULT_TASK_PARAMS.copy()
        task_params_final.update(task_params)
        
        for name, value in task_params_final.items():
            setattr(self, name, value)

        self.device = device
        self.stride = stride

        self.verbose=verbose

        # initialize task specific layers
        self._add_layers()
    
    def _add_layers(self):
        D = 400
        sample = next(self.dataset.__iter__())
        C, T = sample.shape
        with torch.no_grad():
            fake_input = torch.randn(1, C, self.win)
            embed_dim = torch.flatten(self.model(fake_input)).shape[0]
        self.linear_ff = nn.Linear(embed_dim, D)
        self.loss_linear = nn.Linear(D, 1)

    def get_task_model_params(self):
        return list(self.linear_ff.parameters()) + list(self.loss_linear.parameters())

    def gTS(self, embeddings):
        differences = []

        for i in range(len(embeddings)):
          differences.append(torch.cat((torch.abs(embeddings[i][0] - embeddings[i][1]), torch.abs(embeddings[i][1] - embeddings[i][2])), dim = 0))

        return torch.stack(differences)

    def forward(self, model, x):
        self.linear_ff.to(self.device).train()
        self.loss_linear.to(self.device).train()
        samples = []
        labels = []

        window_nsample = int(self.sfreq * self.win)
        tau_pos_nsample = self.sfreq * self.tau_pos
        tau_neg_nsample = self.sfreq * self.tau_neg
        tau_pos = self.tau_pos

        for pos_start in np.arange(0, x.shape[2]-window_nsample, tau_pos_nsample): # non-overlapping positive contexts
                pos_winds = [x[:, :, pos_start                                : pos_start+window_nsample], 
                             x[:, :, pos_start+tau_pos_nsample-window_nsample : pos_start+tau_pos_nsample]] # two positive windows\
                inorder_start = np.random.choice(np.arange(pos_start+window_nsample, pos_start+tau_pos_nsample-window_nsample, window_nsample), 1)
                inorder_window = x[:, :, inorder_start:inorder_start+window_nsample]
                samples.append(pos_winds + inorder_window)
                labels.append(torch.ones(1))

                # for negative windows, want both sides of anchor window
                tau_neg_before = np.arange(0, anchor_start - tau_neg_nsample - window_nsample, window_nsample)
                tau_neg_after = np.arange(anchor_start + window_nsample + tau_neg_nsample, x.shape[2]-window_nsample, window_nsample)
                if len(tau_neg_before) > 0 or len(tau_neg_after) > 0:
                    for neg_wind_start in np.random.choice(np.concatenate([tau_neg_before, tau_neg_after]), self.n_samples, replace=False):
                        # print('neg_wind_start', neg_wind_start)
                        anch  = x[:, :,anchor_start:anchor_start+window_nsample]
                        neg_w = x[:, :,neg_wind_start:neg_wind_start+window_nsample]

                        samples.append(torch.stack([anch, neg_w])) # if anchors[i].shape == pos_winds[i].shape])
                        labels.append(torch.zeros(1))

                neg_winds_start = np.concatenate((np.arange(0, pos_start-self.tau_neg-self.win, self.stride), np.arange(pos_start+tau_pos+self.tau_neg, x.shape[2]-self.win, self.stride)))
                selected_neg_start = np.random.choice(neg_winds_start, 1, replace=False)[0]
                disorder = torch.stack(pos_winds[:1] + [x[:,:,selected_neg_start:selected_neg_start+self.win]] + pos_winds[1:]) # two positive windows, disorder sample added to the end
                samples.extend([disorder, torch.flip(disorder, dims = [0])])
                labels.extend(torch.zeros(2))

        samples = torch.stack(samples)
        labels = torch.stack(labels).unsqueeze(1)
        if len(samples) != len(labels):
            raise ValueError('Number of samples and labels mismatch')

        samples = samples.to(device=self.device, dtype=torch.float32)
        embeddings = []
        for i in range(samples.shape[0]):
          embeddings.append(self.linear_ff(model(samples[i][:, 0])))

        differences = self.gTS(embeddings)
        labels = labels.long()

        loss = self.criterion(differences, labels)

        del differences
        del labels

        return loss

    def loss(self, differences, labels):
        linear_combination = self.loss_linear(differences)
        # Calculate the loss
        loss = torch.log(1 + torch.exp(-labels * linear_combination))
        return loss.mean()

class Trainer():
    # default training parameters
    DEFAULT_TRAIN_PARAMS = {
        'num_epochs': 100,
        'batch_size': 10,
        'print_every': 100,
        'num_workers': 0,
    }

    def __init__(self, 
                dataset: torch.utils.data.IterableDataset,
                model: torch.nn.Module,
                task_params={},
                train_params=DEFAULT_TRAIN_PARAMS,
                wandb=None,
                verbose=False,
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 
                seed=9,
        ):
        self.dataset = dataset
        self.model = model
        self.device = device
        self.verbose = verbose
        self.wandb = wandb

        self.seed = seed
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)

        
        train_params_final = self.DEFAULT_TRAIN_PARAMS.copy()
        train_params_final.update(train_params)
        print('Training parameters', train_params_final)
        print('Task parameters', task_params)

        for name, value in train_params_final.items():
            setattr(self, name, value)
    
        # initialize task instance using task name and task_params
        self.task = getattr(globals()[task_params['task']], '__call__')(self.dataset, self.model, task_params, device=self.device, verbose=verbose)

        self.optimizer = torch.optim.Adam(list(model.parameters()) + self.task.get_task_model_params())

    def train(self, checkpoint=None):
        print('Training on ', self.device)
        task = self.task
        dataset = self.dataset
        model = self.model
        optimizer = self.optimizer
        num_epochs = self.num_epochs
        batch_size = self.batch_size
        print_every = self.print_every
        num_workers = self.num_workers
        wandb = self.wandb
        epoch_start = 0

        if self.verbose:
            print('Training with parameters:')
            for name, value in locals().items():
                print(f'{name}: {value}')


        # resume from checkpoint if provided
        if checkpoint is not None and os.path.exists(checkpoint):
            checkpoint_dict = torch.load(checkpoint, weights_only=True)
            print('Resuming from checkpoint ', checkpoint, ' at epoch ', checkpoint_dict['epoch'])
            model.load_state_dict(checkpoint_dict['model_state_dict'])
            self.task.task_module.load_state_dict(checkpoint_dict['task_module_state_dict'])
            optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
            epoch_start = checkpoint_dict['epoch'] + 1 # assuming checkpoint is saved at the end of each epoch

        # Move the model and optimizer state to the desired device
        model.to(device=self.device)
        model.train()
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

        if wandb and wandb.run:
            wandb.watch(model, log='all', log_freq=100)

        for e in range(epoch_start, num_epochs):
            dataset._shuffle()
            dataloader_train = DataLoader(dataset, batch_size = batch_size, num_workers=num_workers)
            for t, samples in enumerate(dataloader_train):
                # check if samples has nan
                assert not np.any(np.isnan(samples.numpy()))

                loss = task.forward(model, samples)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if t % print_every == 0:
                    # writer.add_scalar("Loss/train", loss.item(), e*len(dataloader)+t)
                    print('Epoch %d, Iteration %d, loss = %.4f' % (e, t, loss.item()))

                metrics = {"train/train_loss": loss.item()}
                if wandb and wandb.run is not None:
                    metrics = {"epoch": e, "iter": t, "train/train_loss": loss.item()}
                    wandb.log(metrics)

                # save checkpoint
                if t % 2999 == 0 and wandb and wandb.run is not None:
                    torch.save({
                        'epoch': e,
                        'iteration': t,
                        'model_state_dict': model.state_dict(),
                        'task_module_state_dict': self.task.task_module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                    }, os.path.join(wandb.run.dir, f"checkpoint_epoch-{e}_iteration-{t}"))

                del samples

            if wandb and wandb.run is not None:
                torch.save({
                    'epoch': e,
                    'model_state_dict': model.state_dict(),
                    'task_module_state_dict': self.task.task_module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, os.path.join(wandb.run.dir, f"checkpoint_epoch-{e}_final"))

        if wandb and wandb.run is not None:
            torch.save({
                'epoch': e,
                'model_state_dict': model.state_dict(),
                'task_module_state_dict': self.task.task_module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, os.path.join(wandb.run.dir, f"checkpoint_final"))
            wandb.finish()
            # eval_train_score, eval_test_score = self.finetune_eval_score(model)

    def finetune_eval_score(self, model):
        model.eval()
        generator = torch.Generator().manual_seed(42)
        val_train, val_test = random_split(self.dataset_val, [0.7, 0.3], generator=generator)
        val_train_dataloader = DataLoader(val_train, batch_size = len(val_train), shuffle = True)
        val_test_dataloader = DataLoader(val_test, batch_size = len(val_test), shuffle = True)

        samples, labels = next(iter(val_train_dataloader))
        samples = samples.to(device=self.device, dtype=torch.float32)
        predictions = model(samples)
        # print(predictions)
        embeddings = torch.mean(predictions, dim=-1) # TODO is averaging the best strategy here, for classification?
        # print(embeddings)
        clf = LinearDiscriminantAnalysis()
        clf.fit(embeddings.detach().cpu().numpy(), labels.detach().cpu().numpy())
        train_score = clf.score(embeddings.detach().cpu().numpy(), labels.detach().cpu().numpy())
        print('Eval train score:', train_score)

        samples_test, labels_test = next(iter(val_test_dataloader))
        samples_test = samples_test.to(device=self.device, dtype=torch.float32)
        predictions = model(samples_test)
        embeddings = torch.mean(predictions, dim=-1) # TODO is averaging the best strategy here, for classification?
        test_score = clf.score(embeddings.detach().cpu().numpy(), labels_test.detach().cpu().numpy())
        print('Eval test score:', test_score)
        return train_score, test_score
        
class CPC():
    def __init__(self,
                dataset: torch.utils.data.Dataset,
                win_length = 50,
                tau_pos = 150,
                tau_neg = 151,
                n_samples = 1,
                stride = 1,
                task_params={
                    'mask_prob': 0.5
                },
                train_params={
                    'num_epochs': 100,
                    'batch_size': 10,
                    'print_every': 10
                },
                verbose=False
        ):
        self.dataset = dataset
        self.train_test_split()
        self.win = win_length
        self.tau_pos = tau_pos
        self.tau_neg = tau_neg
        self.n_samples = n_samples
        self.stride = stride
        self.Nc = 5
        self.Np = 2
        self.Nb = 2
        self.linear_ff = None

        self.train_params = train_params
        self.mask_probability = task_params['mask_prob']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.verbose=verbose
        self.gru = nn.GRU(200, 100)
        self.linear_gAR = nn.Linear(100*self.Nc, 100)
        self.linear_fk = nn.Linear(100, 200)

    def gAR(self, x):
        x = self.gru(x)[0]
        x = torch.flatten(x)
        return self.linear_gAR(x)

    def fk(self, h, c):
        x = self.linear_fk(c)
        return torch.matmul(h, x)

    def train_test_split(self):
        generator = torch.Generator().manual_seed(42)
        self.dataset_train, self.dataset_val = torch.utils.data.random_split(self.dataset, [0.7,0.3], generator=generator)

    def forward(self, model, x, opt):
        context_windows_all = []
        future_windows_all = []
        negative_windows_all = []
        ct = []
        for ti in np.arange(0, x.shape[2]-self.win*(self.Nc+self.Np), self.win*self.Nc):
            context_windows = []
            # ti is the index of the first window in the context array
            for st in np.arange(ti, ti+(self.win*self.Nc), self.win):
              context_windows.append(x[:, :, st:st+self.win])
            c_w = torch.stack(context_windows)
            context_windows_all.append(c_w)

            if self.linear_ff is None:
                self.linear_ff = nn.Linear(torch.flatten(model.feature_encoder(c_w[:, 0]), start_dim = 1).shape[1], 200)
                opt.add_param_group({'params': list(self.linear_ff.parameters())})

            embeddings = self.linear_ff(torch.flatten(model.feature_encoder(c_w[:, 0]), start_dim = 1))
            ct.append(self.gAR(embeddings))

            future_windows = []
            for st in np.arange(ti+(self.win*self.Nc), ti+(self.win*(self.Nc+self.Np)), self.win):
              future_windows.append(x[:, :, st:st+self.win])
            future_windows_all.append(torch.stack(future_windows))

            n_w = []
            for i in range(len(future_windows)):
                negative_windows = []
                for j in range(self.Nb):
                  st = np.random.choice(list(range(0, ti)) + list(range(ti+(self.win*(self.Nc+self.Np)), x.shape[2]-self.win)), replace=False)
                  negative_windows.append(x[:, :, st:st+self.win])
                n_w.append(torch.stack(negative_windows))
            negative_windows_all.append(torch.stack(n_w))

        future_windows_all = torch.stack(future_windows_all)
        future_embeddings = []
        for i in range(future_windows_all.shape[0]):
            future_embeddings.append(self.linear_ff(torch.flatten(model.feature_encoder(future_windows_all[i, :, 0]), start_dim = 1)))

        negative_windows_all = torch.stack(negative_windows_all)
        negative_embeddings = []
        for i in range(negative_windows_all.shape[0]):
            negative_embeddings.append([])
            for j in range(negative_windows_all.shape[1]):
                negative_embeddings[i].append(self.linear_ff(torch.flatten(model.feature_encoder(negative_windows_all[i, j, :, 0]), start_dim = 1)))
            negative_embeddings[i] = torch.stack(negative_embeddings[i])

        return torch.stack(ct), torch.stack(future_embeddings), torch.stack(negative_embeddings)

    def loss(self, ct, future_embeddings, negative_embeddings):
        loss = 0
        for i in range(ct.shape[0]):
            for j in range(future_embeddings.shape[1]):
                dot_negative = 0
                for k in range(negative_embeddings.shape[2]):
                    dot_negative += torch.exp(self.fk(negative_embeddings[i, j, k], ct[i]))
                den = dot_negative + torch.exp(self.fk(future_embeddings[i, j], ct[i]))
                num = torch.exp(self.fk(future_embeddings[i, j], ct[i]))
                loss -= torch.log(num/den)
        return loss

