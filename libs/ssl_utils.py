import torch
from torch.nn import functional as F
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from joblib import Parallel, delayed
import os
import mne
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class MaskedContrastiveLearningTask():
    def __init__(self, 
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
        self.verbose=verbose
        self.train_params = train_params
        self.mask_probability = task_params['mask_prob']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        print(model.feature_encoder.device)
        print(model.context_encoder.device)
        embeddings = model.feature_encoder(x) # N x D x K
        if self.verbose:
            print('feature encoder output shape', embeddings.shape)

        # learned masked vector embedding
        masked_vector_learned_embedding = torch.ones((embeddings.shape[0], embeddings.shape[1])) # N x D # TODO
        if self.verbose:
            print('learned masked embeddings shape', masked_vector_learned_embedding.shape)

        # select from the sampled segment L masked inputs
        masked_indices = np.random.choice(embeddings.shape[-1], size=(int(self.mask_probability*embeddings.shape[-1]),), replace=False)
        if self.verbose:
            print('masked indices shape', masked_indices.shape)
        # replace the selected indices with the masked vector embedding
        true_masked_embeddings = embeddings[:,:,masked_indices] # N x D x K # .detach().clone() 
        if self.verbose:
            print('true masked embeddings shape', true_masked_embeddings.shape)

        learned_embeddings_replace = embeddings.clone() # if not clone backward pass will complain as inplace modification not allowed
        for i in range(len(masked_indices)):
            learned_embeddings_replace[:,:,i] = masked_vector_learned_embedding
        if self.verbose:
            print('masked embeddings shape', embeddings.shape)

        # feed masked samples to context encoder. Every timestep has an output
        context_encoder_outputs = model.context_encoder(learned_embeddings_replace) # N x D x K
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
        losses = torch.zeros((masked_latents.shape[-1],))
        for k in range(masked_latents.shape[-1]):
            predicted_masked_latent = predictions[:,:,k] # N x D
            if self.verbose:
                print('predicted masked latent shape', predicted_masked_latent.shape)
            cos_sim = F.cosine_similarity(torch.unsqueeze(predicted_masked_latent, dim=-1), masked_latents, dim=1) # N x K
            if self.verbose:
                print('cosine similarity shape', cos_sim.shape)
            labels = torch.zeros([cos_sim.shape[0], cos_sim.shape[1]]) # N x K
            labels[:,k] = 1
            # print('labels', labels)
            # losses.append(F.cross_entropy(cos_sim, labels, reduction='mean'))
            losses[k] = F.cross_entropy(cos_sim, labels, reduction='mean')
        if self.verbose:
            print('losses', losses)
        # return torch.mean(torch.tensor(losses))
        return torch.mean(losses)

    def train(self, model, dataset_train, dataset_val, train_params={}):
        self.train_params.update(train_params)
        num_epochs = self.train_params['num_epochs']
        batch_size = self.train_params['batch_size']
        print_every = self.train_params['print_every']

        optimizer  = torch.optim.Adam(model.parameters())
        dataloader_train = DataLoader(dataset_train, batch_size = batch_size, shuffle = True)
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

                del samples
                del predictions
                del masked_latents
                del loss

            model.eval()
            generator = torch.Generator().manual_seed(42)
            val_train, val_test = random_split(dataset_val, [0.7, 0.3], generator=generator)
            val_train_dataloader = DataLoader(val_train, batch_size = batch_size, shuffle = True)
            for t, (samples, labels) in enumerate(val_train_dataloader):
                samples = samples.to(device=self.device, dtype=torch.float32)
                predictions = model(samples)
                embeddings = torch.mean(predictions, dim=-1) # TODO is averaging the best strategy here, for classification?
                clf = LinearDiscriminantAnalysis()
                clf.fit(embeddings.to_numpy(), labels.to_numpy())

                score = clf.score(embeddings.)




