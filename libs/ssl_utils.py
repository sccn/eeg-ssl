import torch
from torch.nn import functional as F
import torchvision.models as torchmodels
import torch.nn as nn
import numpy as np
import ssl_dataloader as ssl_dataloader
from torch.utils.data import Dataset, DataLoader, random_split

class SSLTask(ABC, nn.Module):
    def __init__(self, task_params=None):
        super().__init__()
        default_params = {
            'task': 'RP',
        }

        if task_params:
            default_params.update(task_params)
        for k,v in default_params.items():
            setattr(self, k, v)

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def loss(self, x):
        pass

    @abstractmethod
    def train(self):
        pass

class MaskedContrastiveLearning(SSLTask):
    def __init__(self, task_params=None):
        super().__init__(task_params)
        self.mask_size = task_params['mask_size']
        self.mask_probability = 0.5

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
    
    def get_dataloader(self, dataset):
        segment = self.SegmentInput(self.mask_size)
        segment_input = segment(dataset)
        dataloader = DataLoader(segment_input, batch_size = self.train_params['batch_size'], shuffle = True)

        return dataloader

    def forward(self, model, x):
        '''
        Forward pass of the model
        @parameter
            model:  nn.Module model
            x:      (N x C x T) batched input
        @return
            prediction:         (N x D) Batch-size embeddings of the model's guess for masked inputs
            masked_latent:      (N x D) Batch-size embeddings of the feature encoder output of true masked inputs
            foil_latents:       (N x K x D) Batch-size embeddings of the feature conder output of the foil inputs
        '''
        sample_segment = self.segment_input(x, self.mask_size) # K x N x C x M
        embeddings = [model.feature_encoder(batch_segment) for batch_segment in sample_segment]
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
        params = SimpleNamespace(**self.train_params)
        writer = SummaryWriter(params.log_dir)
        dataloader = self.get_dataloader(dataset)
        for e in range(params.num_epochs):
            for t, (samples, _) in enumerate(dataloader):
                predictions, masked_latents = self.forward(model, samples)
                loss = self.loss(predictions, masked_latents)
                params.optimizer.zero_grad()
                loss.backward()
                params.optimizer.step()

                if t % params.print_every == 0:
                    writer.add_scalar("Loss/train", loss.item(), e*len(dataloader)+t)
                    print('Epoch %d, Iteration %d, loss = %.4f' % (e, t, loss.item()))

                del predictions
                del masked_latents
                del loss

    class SegmentInput(object):
        def __init__(self, M):
            assert isinstance(M, int)
            self.M = M

        def __call__(self, x):    
            '''
            Split input into segments of M time sample
            @parameter
                x: (N x C x T) Batched multichannel EEG input
            @return
                output  (K x N x C x M) Batched multichannel EEG input, each having K segments
            '''
            # sample from left to right, discarding leftovers
            indices = np.arange(0, x.shape[-1]-self.M, self.M)
            samples = [x[:,:,idx:idx+self.M] for idx in indices]
            samples = np.stack(samples, axis=0)

            return samples
