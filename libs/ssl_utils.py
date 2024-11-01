import torch
from torch.nn import functional as F
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from joblib import Parallel, delayed
import os
#import wandb
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import itertools
from abc import ABC, abstractmethod

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
        self.train_test_split()

    def train_test_split(self):
        generator = torch.Generator().manual_seed(self.seed)
        self.dataset_train, self.dataset_val = torch.utils.data.random_split(self.dataset, [0.7,0.3], generator=generator)

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
        dataloader_train = DataLoader(self.dataset_train, batch_size = batch_size, shuffle = True)
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

class SSLTask(ABC):
    def __init__(self, dataset, model):
        self.dataset = dataset
        self.model = model
    
    @abstractmethod
    def _add_layers(self, model, opt):
        pass

    @abstractmethod
    def forward(self, model, samples, opt):
        pass

    @abstractmethod
    def loss(self, outputs):
        pass

class RelativePositioningTask(SSLTask):
    def __init__(self,
                task_params={
                    'win_length': 30,
                    'tau_pos': 75,
                    'tau_neg': 85,
                    'n_samples': 1,
                    'emb_length': 200,
                    'channels': 128
                },
                verbose=False
        ):
        self.win = task_params['win_length']
        self.tau_pos = task_params['tau_pos']
        self.tau_neg = task_params['tau_neg']
        self.n_samples = task_params['n_samples']
        self.emb_length = task_params['emb_length']
        self.C = task_params['channels']

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.verbose=verbose
        self.loss_linear_ff = None

    def gRP(self, embeddings):
        differences = []

        for i in range(len(embeddings)):
          differences.append(torch.abs(embeddings[i][0] - embeddings[i][1]))

        return torch.stack(differences)

    def _add_layers(self, model, opt):
        D = self.emb_length
        with torch.no_grad():
            fake_input = torch.randn(1, self.C, self.win).to(self.device)
            embed_dim = torch.flatten(model(fake_input)).shape[0]
        self.linear_ff = nn.Linear(embed_dim, D)
        self.linear_ff.to(self.device)
        opt.add_param_group({'params': list(self.linear_ff.parameters())})
        self.loss_linear = nn.Linear(D, 1)
        self.loss_linear.to(self.device)
        opt.add_param_group({'params': list(self.loss_linear.parameters())})

    def forward(self, model, x, opt):
        samples = []
        labels = []
        
        if self.loss_linear_ff is None:
          self._add_layers(model, opt)

        for anchor_start in np.arange(0, x.shape[2]-self.win, self.win): # non-overlapping anchor window
            # Positive window start t_pos:
            #     - |t_pos - t_anchor| <= tau_pos
            #           <-> t_pos <= tau_pos + t_anchor
            #           <-> t_pos => t_anchor - tau_pos
            #     - t_pos < T - win
            #.    - t_pos > 0
            pos_winds_start = np.arange(np.maximum(0, anchor_start - self.tau_pos), np.minimum(anchor_start+self.tau_pos, x.shape[2]-self.win), self.win) # valid positive samples onsets
            if len(pos_winds_start) > 0:
                # positive context
                pos_winds = [x[:, :, sample_start:sample_start+self.win] for sample_start in np.random.choice(pos_winds_start, self.n_samples, replace=False)]
                anchors = [x[:, :,anchor_start:anchor_start+self.win] for i in range(len(pos_winds))] # repeat same anchor window

                anch = torch.stack([anchors[i].clone().detach() for i in range(len(anchors))])[0]
                pos_w = torch.stack([pos_winds[i].clone().detach() for i in range(len(anchors))])[0]

                samples.append(torch.stack([anch, pos_w])) # if anchors[i].shape == pos_winds[i].shape])
                labels.append(torch.ones(len(anchors)))

                # negative context
                # Negative window start t_neg:
                #     - |t_neg - t_anchor| > tau_neg
                #           <-> t_neg > tau_neg + t_anchor
                #           <-> t_neg < t_anchor - tau_neg
                #     - t_neg < T - win
                #.    - t_neg > 0
                neg_winds_start = np.concatenate((np.arange(0, anchor_start-self.tau_neg, self.win), np.arange(anchor_start+self.tau_neg, x.shape[2]-self.win, self.win)))
                neg_winds = [x[:, :,sample_start:sample_start+self.win] for sample_start in np.random.choice(neg_winds_start, self.n_samples, replace=False)]

                anch = torch.stack([anchors[i].clone().detach() for i in range(len(anchors))])[0]
                neg_w = torch.stack([neg_winds[i].clone().detach() for i in range(len(anchors))])[0]

                samples.append(torch.stack([anch, neg_w])) # if anchors[i].shape == neg_winds[i].shape])
                labels.append(torch.zeros(len(anchors)))

        samples = torch.stack(samples) # N x 2 (anchors, pos/neg) x C x W
        if len(samples) != len(labels):
            raise ValueError('Number of samples and labels mismatch')
        labels = torch.stack(labels)
        labels = labels.to(self.device)

        embeddings = []
        if self.linear_ff is None:
            self.linear_ff = nn.Linear(torch.flatten(model.feature_encoder(samples[0][:, 0]), start_dim = 1).shape[1], 200)
            opt.add_param_group({'params': list(self.linear_ff.parameters())})

        for i in range(samples.shape[0]):
            embeddings.append(self.linear_ff(torch.flatten(model(samples[i][:, 0]), start_dim = 1)))

        differences = self.gRP(embeddings)
        labels = labels.long()

        return [differences, labels]

    def loss(self, outputs):
        differences = outputs[0]
        labels = outputs[1]
        linear_combination = self.loss_linear(differences)
        # Calculate the loss
        loss = torch.log(1 + torch.exp(-labels * linear_combination))
        return loss.mean()
        
class TemporalShufflingTask(SSLTask):
    def __init__(self,
                task_params={
                    'win_length': 30,
                    'tau_pos': 90,
                    'tau_neg': 100,
                    'n_samples': 1,
                    'stride': 1,
                    'emb_length': 200,
                    'channels': 128
                },
                verbose=False
        ):
        self.win = task_params['win_length']
        self.tau_pos = task_params['tau_pos']
        self.tau_neg = task_params['tau_neg']
        self.n_samples = task_params['n_samples']
        self.stride = task_params['stride']
        self.emb_length = task_params['emb_length']
        self.C = task_params['channels']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.verbose=verbose
        self.loss_linear_ff = None

    def gTS(self, embeddings):
        differences = []

        for i in range(len(embeddings)):
          differences.append(torch.cat((torch.abs(embeddings[i][0] - embeddings[i][1]), torch.abs(embeddings[i][1] - embeddings[i][2])), dim = 0))

        return torch.stack(differences)        

    def _add_layers(self, model, opt):
        D = self.emb_length
        with torch.no_grad():
            fake_input = torch.randn(1, self.C, self.win).to(self.device)
            embed_dim = torch.flatten(model(fake_input)).shape[0]
        self.linear_ff = nn.Linear(embed_dim, D)
        self.linear_ff.to(self.device)
        opt.add_param_group({'params': list(self.linear_ff.parameters())})
        self.loss_linear = nn.Linear(2*D, 1)
        self.loss_linear.to(self.device)
        opt.add_param_group({'params': list(self.loss_linear.parameters())})

    def forward(self, model, x, opt):
        samples = []
        labels = []
        
        if self.loss_linear_ff is None:
          self._add_layers(model, opt)

        tau_pos = self.tau_pos
        for pos_start in np.arange(0, x.shape[2]-self.win, self.win): # non-overlapping positive contexts
            if pos_start + tau_pos < x.shape[2]:
                pos_winds = [x[:, :, pos_start:pos_start+self.win], x[:, :, pos_start+self.win*2:pos_start+self.win*3]] # two positive windows\
                inorder = torch.stack(pos_winds[:1] + [x[:, :, pos_start+self.win:pos_start+self.win*2]] + pos_winds[1:])
                samples.extend([inorder, torch.flip(inorder, dims = [0])])
                labels.extend(torch.ones(2))

                # for negative windows, want both sides of anchor window
                buffer = int((self.tau_neg - self.tau_pos)/2)
                neg_winds_start = np.concatenate((np.arange(0, pos_start-buffer-self.win, self.stride), np.arange(pos_start+tau_pos+buffer, x.shape[2]-self.win, self.stride)))
                selected_neg_start = np.random.choice(neg_winds_start, 1, replace=False)[0]
                disorder = torch.stack(pos_winds[:1] + [x[:,:,selected_neg_start:selected_neg_start+self.win]] + pos_winds[1:]) # two positive windows, disorder sample added to the end
                samples.extend([disorder, torch.flip(disorder, dims = [0])])
                labels.extend(torch.zeros(2))

        samples = torch.stack(samples)
        labels = torch.stack(labels).unsqueeze(1)
        if len(samples) != len(labels):
            raise ValueError('Number of samples and labels mismatch')

        embeddings = []

        if self.linear_ff is None:
            self.linear_ff = nn.Linear(torch.flatten(model(samples[0][:, 0]), start_dim = 1).shape[1], 200)
            self.linear_ff.to(self.device)
            opt.add_param_group({'params': list(self.linear_ff.parameters())})

        for i in range(samples.shape[0]):
          embeddings.append(self.linear_ff(torch.flatten(model(samples[i][:, 0]), start_dim = 1)))

        differences = self.gTS(embeddings)
        labels = labels.long()
        labels = labels.to(self.device)

        return [differences, labels]

    def loss(self, outputs):
        differences = outputs[0]
        labels = outputs[1]
        linear_combination = self.loss_linear(differences)
        # Calculate the loss
        loss = torch.log(1 + torch.exp(-labels * linear_combination))
        return loss.mean()

class CPC(SSLTask):
    def __init__(self,
                task_params={
                    'win_length': 30,
                    'tau_pos': 150,
                    'tau_neg': 151,
                    'n_samples': 1,
                    'Nc': 2,
                    'Np': 2,
                    'Nb': 2,
                    'emb_length': 200,
                    'channels': 128
                },
                verbose=False
        ):
        self.win = task_params['win_length']
        self.tau_pos = task_params['tau_pos']
        self.tau_neg = task_params['tau_neg']
        self.n_samples = task_params['n_samples']
        self.Nc = task_params['Nc']
        self.Np = task_params['Np']
        self.Nb = task_params['Nb']
        self.emb_length = task_params['emb_length']
        self.C = task_params['channels']
        self.linear_ff = None

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.verbose=verbose

    def gAR(self, x):
        x = self.gru(x)[0]
        x = torch.flatten(x)
        return self.linear_gAR(x)

    def fk(self, h, c):
        x = self.linear_fk(c)
        return torch.matmul(h, x)

    def _add_layers(self, model, opt):
        D = self.emb_length
        with torch.no_grad():
            fake_input = torch.randn(1, self.C, self.win).to(self.device)
            embed_dim = torch.flatten(model(fake_input)).shape[0]
        self.linear_ff = nn.Linear(embed_dim, D)
        self.linear_ff.to(self.device)
        opt.add_param_group({'params': list(self.linear_ff.parameters())})
        self.gru = nn.GRU(D, 100)
        self.gru.to(self.device)
        opt.add_param_group({'params': list(self.gru.parameters())})
        self.linear_gAR = nn.Linear(100*self.Nc, 100)
        self.linear_gAR.to(self.device)
        opt.add_param_group({'params': list(self.linear_gAR.parameters())})
        self.linear_fk = nn.Linear(100, 200)
        self.linear_fk.to(self.device)
        opt.add_param_group({'params': list(self.linear_fk.parameters())})

    def forward(self, model, x, opt):
        context_windows_all = []
        future_windows_all = []
        negative_windows_all = []
        ct = []

        if self.linear_ff is None:
          self._add_layers(model, opt)

        for ti in np.arange(0, x.shape[2]-self.win*(self.Nc+self.Np), self.win*self.Nc):
            context_windows = []
            # ti is the index of the first window in the context array
            for st in np.arange(ti, ti+(self.win*self.Nc), self.win):
              context_windows.append(x[:, :, st:st+self.win])
            c_w = torch.stack(context_windows)
            context_windows_all.append(c_w)

            if self.linear_ff is None:
                self.linear_ff = nn.Linear(torch.flatten(model(c_w[:, 0]), start_dim = 1).shape[1], 200)
                self.linear_ff.to(self.device)
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

        return [torch.stack(ct), torch.stack(future_embeddings), torch.stack(negative_embeddings)]

    def loss(self, outputs):
        ct = outputs[0]
        future_embeddings = outputs[1]
        negative_embeddings = outputs[2]
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

class Trainer():
    # default training parameters
    DEFAULT_TRAIN_PARAMS = {
        'num_epochs': 100,
        'batch_size': 1,
        'print_every': 100,
        'num_workers': 0,
    }

    def __init__(self, 
                train_dataset: torch.utils.data.Dataset,
                val_dataset: torch.utils.data.Dataset,
                model: torch.nn.Module,
                task,
                train_params=DEFAULT_TRAIN_PARAMS,
                wandb=None,
                verbose=False,
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        ):
        self.train_params = train_params
        self.dataset_train = train_dataset
        self.dataset_val = val_dataset
        self.model = model
        self.device = device
        self.verbose = verbose
        self.wandb = wandb
        
        train_params_final = self.DEFAULT_TRAIN_PARAMS.copy()
        train_params_final.update(train_params)
        print('Training parameters', train_params_final)

        for name, value in train_params_final.items():
            setattr(self, name, value)
    
        # initialize task instance using task name and task_params
        self.task = task
        self.optimizer = torch.optim.Adam(list(model.parameters()))

    def train(self):
        print('Training on ', self.device)
        num_epochs = self.train_params['num_epochs']
        batch_size = self.train_params['batch_size']
        print_every = self.train_params['print_every']

        dataloader_train = DataLoader(self.dataset_train, batch_size = batch_size, shuffle = True)
        model = self.model
        model.to(device=self.device)
        model.train()
        task = self.task
        for e in range(num_epochs):
            for t, (samples, _) in enumerate(dataloader_train):
                samples = samples.to(device=self.device, dtype=torch.float32)
                outputs = task.forward(model, samples, self.optimizer)
                loss = task.loss(outputs)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if t % print_every == 0:
                    # writer.add_scalar("Loss/train", loss.item(), e*len(dataloader)+t)
                    print('Epoch %d, Iteration %d, loss = %.4f' % (e, t, loss.item()))

                metrics = {"train/train_loss": loss.item()}

                for i in range(len(outputs)):
                  del outputs[0]
                del samples
                del loss

            eval_train_score, eval_test_score = self.finetune_eval_score(model)

    def finetune_eval_score(self, model):
        model.eval()
        generator = torch.Generator().manual_seed(42)
        val_train, val_test = random_split(self.dataset_val, [0.7, 0.3], generator=generator)
        val_train_dataloader = DataLoader(val_train, batch_size = 4, shuffle = True)
        val_test_dataloader = DataLoader(val_test, batch_size = 4, shuffle = True)
        
        train_score = 0
        for samples, labels in val_train_dataloader:
            samples = samples.to(device=self.device, dtype=torch.float32)
            predictions = F.normalize(model(samples))
            # print(predictions)
            embeddings = torch.mean(predictions, dim=-1) # TODO is averaging the best strategy here, for classification?
            # print(embeddings)
            clf = LinearDiscriminantAnalysis()
            clf.fit(embeddings.detach().cpu().numpy(), labels.detach().cpu().numpy())
            train_score += clf.score(embeddings.detach().cpu().numpy(), labels.detach().cpu().numpy())
        print('Eval train score:', train_score)

        test_score = 0
        for samples_test, labels_test in val_test_dataloader:
            samples_test = samples_test.to(device=self.device, dtype=torch.float32)
            predictions = F.normalize(model(samples_test))
            embeddings = torch.mean(predictions, dim=-1) # TODO is averaging the best strategy here, for classification?
            test_score += clf.score(embeddings.detach().cpu().numpy(), labels_test.detach().cpu().numpy())
        print('Eval test score:', test_score)
        return train_score, test_score
