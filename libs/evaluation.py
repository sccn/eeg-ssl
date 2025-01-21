import sys
sys.path.insert(0, '/home/dung/eeg-ssl')
import os
from braindecode.datautil import load_concat_dataset
from braindecode.datasets import BaseConcatDataset
from braindecode.preprocessing import create_fixed_length_windows
import torch
import torch.nn as nn
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat
import numpy as np

class RankMe(Metric):
    '''
    From paper: 
    Garrido, Q., Balestriero, R., Najman, L. & Lecun, Y. RankMe: Assessing the downstream performance of pretrained self-supervised representations by their rank. 
    Preprint at https://doi.org/10.48550/arXiv.2210.02885 (2023).
    '''        
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("embs", default=[], dist_reduce_fx='cat')

    def update(self, embs: torch.Tensor) -> None:
        self.embs.append(embs)

    def compute(self) -> torch.Tensor:
        # parse inputs
        embs = dim_zero_cat(self.embs).float()
        if len(embs.shape) > 2:
            raise ValueError('Expect 2D embeddings of shape (N, K)')
        if embs.shape[0] < embs.shape[1]:
            raise ValueError(f'Expect N >= K but received ({embs.shape})')
        _, S, _ = torch.linalg.svd(embs)
        eps = 1e-7
        p = S/torch.linalg.norm(S, ord=1) + eps
        rank_z = torch.exp(-torch.sum(p*torch.log(p)))

        return rank_z

def get_embs(data_path, checkpoint, encoder, save_dir=None):
    if save_dir and os.path.exists(save_dir + '/embeddings.npy'):
        embs = np.load(save_dir + '/embeddings.npy')
    else:
        windows_ds = load_concat_dataset(path=data_path, preload=False)
        emb_size = 100
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        class ContrastiveNet(nn.Module):
            """Contrastive module with linear layer on top of siamese embedder.

            Parameters
            ----------
            emb : nn.Module
                Embedder architecture.
            emb_size : int
                Output size of the embedder.
            dropout : float
                Dropout rate applied to the linear layer of the contrastive module.
            """
            def __init__(self, emb, emb_size, dropout=0.5):
                super().__init__()
                self.emb = emb
                self.pooling = nn.AdaptiveAvgPool2d(32)
                self.clf = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(1024, emb_size),
                    nn.Dropout(dropout),
                    nn.Linear(emb_size, 1)
                )

            def forward(self, x):
                x1, x2 = x
                z1, z2 = self.emb(x1), self.emb(x2)
                z = self.pooling(torch.abs(z1 - z2)).flatten(start_dim=1)

                return self.clf(z).flatten()

        model = ContrastiveNet(encoder, emb_size).to(device)
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()

        dataloader = torch.utils.data.DataLoader(windows_ds, batch_size=1, shuffle=False)
        embs = np.array([])
        for sample in dataloader:
            X = sample[0]
            if embs.shape[0] > 0:
                embs = np.concatenate([embs, model.clf[1](model.pooling(model.emb(X.to('cuda'))).flatten(start_dim=1)).detach().cpu().numpy()])
            else:
                embs = model.clf[1](model.pooling(model.emb(X.to('cuda'))).flatten(start_dim=1)).detach().cpu().numpy()
            # embs = np.concatenate([embs, model.clf[1](model.pooling(model.emb(X.to('cuda'))).flatten(start_dim=1)).detach().cpu().numpy()])
        
        if save_dir:
            np.save(save_dir + '/embeddings.npy', embs)
    return embs

def get_labels(target_name, data_path):
    if type(data_path) == str:
        loaded_dataset = load_concat_dataset(path=data_path, preload=False)
    elif type(data_path) == BaseConcatDataset:
        loaded_dataset = data_path
    return loaded_dataset.get_metadata()[target_name].values

def rankme(embs):
    if len(embs.shape) > 2:
        raise ValueError('Expect 2D embeddings of shape (N, K)')
    if embs.shape[0] < embs.shape[1]:
        raise ValueError('Expect N >= K')
    _, S, _ = np.linalg.svd(embs)
    eps = 1e-7
    p = S/np.linalg.norm(S, ord=1) + eps
    rank_z = np.exp(-np.sum(p*np.log(p)))

    return rank_z

def train_regressor(regr, embs, labels):
    isnan = np.isnan(labels)
    X, Y= embs[~isnan], labels[~isnan]
    assert X.shape[0] == Y.shape[0]
    regr.fit(X, Y)
    score = regr.score(X, Y) 
    return regr, score

def get_prediction_for_subject(subject, embs, labels, regr, subjects):
    subject_embs = embs[subjects==subject]
    subject_labels = labels[subjects==subject]
    return regr.predict(subject_embs).mean(), subject_labels.mean()

def subject_level_score(embs, labels, regr, subjects):
    unique_subjects = np.unique(subjects)
    subject_level_predictions = np.array([get_prediction_for_subject(subject, embs, labels, regr, subjects) for subject in unique_subjects])
    res_sum = ((subject_level_predictions[1] - subject_level_predictions[0])**2).sum()
    total_sum = ((subject_level_predictions[1] - subject_level_predictions[1].mean())**2).sum()
    
    return 1-res_sum/total_sum
