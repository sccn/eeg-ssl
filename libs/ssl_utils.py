import torch
from torch.nn import functional as F
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from joblib import Parallel, delayed
import os
import mne
import wandb
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class MaskedContrastiveLearningTask():
    def __init__(self, 
                dataset: torch.utils.data.Dataset,
                task_params={
                    'mask_prob': 0.5
                },
                train_params={
                    'num_epochs': 100,
                    'batch_size': 10,
                    'print_every': 10
                },
                random_seed=9,
                verbose=False
        ):
        torch.manual_seed(random_seed)
        self.dataset = dataset
        self.train_test_split()

        self.masked_vector_learned_embedding = None
        self.train_params = train_params
        self.mask_probability = task_params['mask_prob']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.verbose=verbose

    def train_test_split(self):
        generator = torch.Generator().manual_seed(42)
        self.dataset_train, self.dataset_val = torch.utils.data.random_split(self.dataset, [0.7,0.3], generator=generator)

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

        # initialize learned masked vector embedding based on dimension D
        if self.masked_vector_learned_embedding is None:
            self.masked_vector_learned_embedding = torch.randn((1, embeddings.shape[1]), requires_grad=True) # N x D # TODO
            if self.verbose:
                print('learned masked embeddings shape', self.masked_vector_learned_embedding.shape)

        # print('masked vector embeddiing', self.masked_vector_learned_embedding.sum())

        # select from the sampled segment L masked inputs
        masked_indices = np.random.choice(embeddings.shape[-1], size=(int(self.mask_probability*embeddings.shape[-1]),), replace=False)
        if self.verbose:
            print('masked indices shape', masked_indices.shape)
        # replace the selected indices with the masked vector embedding
        true_masked_embeddings = embeddings[:,:,masked_indices].detach().clone() # N x D x K 
        if self.verbose:
            print('true masked embeddings shape', true_masked_embeddings.shape)

        for i in range(len(masked_indices)):
            embeddings[:,:,i] = self.masked_vector_learned_embedding.repeat(embeddings.shape[0], 1)
        if self.verbose:
            print('masked embeddings shape', embeddings.shape)

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
        num_epochs = self.train_params['num_epochs']
        batch_size = self.train_params['batch_size']
        print_every = self.train_params['print_every']

        optimizer  = torch.optim.Adam(model.parameters())
        dataloader_train = DataLoader(self.dataset_train, batch_size = batch_size, shuffle = True)
        model.to(device=self.device)
        model.train()
        for e in range(num_epochs):
            for t, (samples, _) in enumerate(dataloader_train):
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

class RelativePositioningTask():
    def __init__(self,
                dataset: torch.utils.data.Dataset,
                win_length = 50,
                tau_pos = 20,
                tau_neg = 30,
                n_samples = 1,
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

        self.train_params = train_params
        self.mask_probability = task_params['mask_prob']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.verbose=verbose

    def train_test_split(self):
        generator = torch.Generator().manual_seed(42)
        self.dataset_train, self.dataset_val = torch.utils.data.random_split(self.dataset, [0.7,0.3], generator=generator)

    def forward(self, model, x):
        samples = []
        labels = []

        for anchor_start in np.arange(0, x.shape[1]-self.win, self.win): # non-overlapping anchor window
            # Positive window start t_pos:
            #     - |t_pos - t_anchor| <= tau_pos
            #           <-> t_pos <= tau_pos + t_anchor
            #           <-> t_pos => t_anchor - tau_pos
            #     - t_pos < T - win
            #.    - t_pos > 0
            pos_winds_start = np.arange(np.maximum(0, anchor_start - self.tau_pos), np.minimum(anchor_start+self.tau_pos, x.shape[1]-self.win), self.win) # valid positive samples onsets
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
                neg_winds_start = np.concatenate((np.arange(0, anchor_start-self.tau_neg, self.win), np.arange(anchor_start+self.tau_neg, x.shape[1]-self.win, self.win)))
                neg_winds = [x[:, :,sample_start:sample_start+self.win] for sample_start in np.random.choice(neg_winds_start, self.n_samples, replace=False)]
                
                anch = torch.stack([anchors[i].clone().detach() for i in range(len(anchors))])[0]
                neg_w = torch.stack([neg_winds[i].clone().detach() for i in range(len(anchors))])[0]

                samples.append(torch.stack([anch, neg_w])) # if anchors[i].shape == neg_winds[i].shape])
                labels.append(torch.zeros(len(anchors)))

        samples = torch.stack(samples) # N x 2 (anchors, pos/neg) x C x W
        if len(samples) != len(labels):
            raise ValueError('Number of samples and labels mismatch')
        labels = torch.stack(labels)

        embeddings = []

        for i in range(samples.shape[0]):
          embeddings.append(model.feature_encoder(samples[i][:, 0]))

        predictions = []

        for i in range(len(embeddings)):
          predictions.append(model.classify(embeddings[i]))

        predictions = torch.stack(predictions)
        labels = labels.long()

        return predictions, labels

    def loss(self, predictions, labels):
        return F.cross_entropy(predictions, labels)

    def train(self, model, train_params={}):
        print('Training on ', self.device)
        self.train_params.update(train_params)
        num_epochs = self.train_params['num_epochs']
        batch_size = self.train_params['batch_size']
        print_every = self.train_params['print_every']

        optimizer  = torch.optim.Adam(model.parameters())
        dataloader_train = DataLoader(self.dataset_train, batch_size = batch_size, shuffle = True)
        model.to(device=self.device)
        model.train()
        for e in range(num_epochs):
            for t, (samples, _) in enumerate(dataloader_train):
                samples = samples.to(device=self.device, dtype=torch.float32)
                predictions, labels = self.forward(model, samples)
                loss = self.loss(predictions, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if t % print_every == 0:
                    # writer.add_scalar("Loss/train", loss.item(), e*len(dataloader)+t)
                    print('Epoch %d, Iteration %d, loss = %.4f' % (e, t, loss.item()))

                metrics = {"train/train_loss": loss.item()}

                del samples
                del predictions
                del labels
                del loss

            eval_train_score, eval_test_score = self.finetune_eval_score(model)

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


