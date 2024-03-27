import torch
from torch.nn import functional as F
import numpy as np

def contrastive_loss(context_output, masked_latent, foil_latents):
    '''
    Follow implementation in https://github.com/dhruvbird/ml-notebooks/blob/main/nt-xent-loss/NT-Xent%20Loss.ipynb
    @parameter
        context_output:     (N x D) Batch-size embeddings of the output of context encoders
        masked_latent:      (N x D) Batch-size embeddings of the feature encoder output of masked inputs
        foil_latents:       (N x K x D) Batch-size embeddings of the feature conder output of the foil inputs
    
    @return
        batched mean contrastive loss
    '''
    # print('K number of foils', foil_latents.shape[1])
    # print('unsquezzed size', torch.unsqueeze(masked_latent, 1).shape)
    embbed_combined = torch.cat([torch.unsqueeze(masked_latent, dim=1), foil_latents], dim=1).permute(0,2,1)
    # print('combined shape permuted shape', embbed_combined.shape) # N x D x K+1
    # print('masked latent', masked_latent[0,:])
    # print('equivalent first element of combined', embbed_combined[0,:,0])
    # print('is equivalent', embbed_combined[0,:,0] == masked_latent[0,:])
    cos_sim = F.cosine_similarity(torch.unsqueeze(context_output, dim=-1), embbed_combined, dim=1)
    # print('cosine similarity', cos_sim)
    labels = torch.zeros([cos_sim.shape[0], cos_sim.shape[-1]])
    labels[:,0] = 1
    # print('labels', labels)
    loss = F.cross_entropy(cos_sim, labels, reduction='mean')
    # print('batch mean loss', loss)

def get_mask_foils(input, sfreq, p, M):
    '''
    From batched input, get mask and foils per sample
    Based on probability p and mask length M
    @parameter
        input:  (N x C x T) batched input
        sfreq: sampling rate of EEG
        M: mask length in samples
        p: percent of timesteps masked
    '''
    indices = np.arange(input.shape[-1])
    indices = indices[:-M]
    # print('indices shape', indices.shape)
    selected_indices = np.concatenate([np.random.choice(indices, size=(1, int(p*input.shape[-1])), replace=False) for i in range(input.shape[0])], axis=0) # uniformly sample without replacement
    # print('selected_indices shape', selected_indices.shape) # should be N x p*T

    masked_indices = np.array([np.random.choice(selected_indices[idx]) for idx in range(selected_indices.shape[0])])
    masks = np.array([input[idx,:,masked_indices[idx]:masked_indices[idx]+M] for idx in range(input.shape[0])])
    # print('masks shape', masks.shape) # N x C x M
    foils  = np.array([[input[sample,:,idx:idx+M] for idx in selected_indices[sample,:] if idx != masked_indices[sample]] for sample in range(input.shape[0])])
    # print('foils shape', foils.shape) # N x K x C x M
                                      # where K = p*N - 1
    return masks, foils