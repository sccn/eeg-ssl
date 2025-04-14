import sys
sys.path.insert(0, '/home/dung/eeg-ssl')
import os
import torch
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

    def update(self, data: tuple) -> None:
        embs = data[0]
        self.embs.append(embs)

    def compute(self) -> torch.Tensor:
        # parse inputs
        embs = dim_zero_cat(self.embs).float()
        if len(embs.shape) > 2:
            raise ValueError('Expect 2D embeddings of shape (N, K)')
        if embs.shape[0] < embs.shape[1]:
            raise ValueError(f'Expect N >= K but received ({embs.shape})')
        # subselect 25600 embeddings randomly
        embs = embs[torch.randperm(embs.shape[0])[:25600]]
        _, S, _ = torch.linalg.svd(embs)
        eps = 1e-7
        p = S/torch.linalg.norm(S, ord=1) + eps
        rank_z = torch.exp(-torch.sum(p*torch.log(p)))

        return rank_z.cpu()

class Regressor(Metric):
    '''
    Validation using regression on target label
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("embs", default=[], dist_reduce_fx='cat')
        self.add_state("labels", default=[], dist_reduce_fx='cat')

    def update(self, data:tuple) -> None:
        embs = data[0]
        labels = data[1]
        self.embs.append(embs)
        self.labels.append(labels)

    def compute(self) -> torch.Tensor:
        from sklearn.linear_model import LinearRegression
        # parse inputs
        embs = dim_zero_cat(self.embs).float()
        labels = dim_zero_cat(self.labels).float()
        if len(embs.shape) > 2:
            raise ValueError('Expect 2D embeddings of shape (N, K)')
        regr = LinearRegression()
        embs = embs.cpu()
        labels = labels.cpu()
        regr.fit(embs, labels)
        score = regr.score(embs, labels) 
        return score

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
