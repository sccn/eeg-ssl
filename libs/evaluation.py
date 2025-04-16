import sys
sys.path.insert(0, '/home/dung/eeg-ssl')
import os
import torch
from torchmetrics import Metric
from torchmetrics.functional.regression import concordance_corrcoef, r2_score, normalized_root_mean_squared_error, mean_squared_error
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
        self.add_state("subjects", default=[], dist_reduce_fx='cat')

    def update(self, data:tuple) -> None:
        embs = data[0]
        labels = data[1]
        subjects = data[2]
        self.embs.append(embs)
        self.labels.append(labels)
        self.subjects.append(subjects)

    def compute(self) -> torch.Tensor:
        from sklearn.linear_model import LinearRegression
        # parse inputs
        embs = dim_zero_cat(self.embs).float()
        labels = dim_zero_cat(self.labels).float()
        # subjects = dim_zero_cat(self.subjects).float()
        # print(self.subjects)
        if len(embs.shape) > 2:
            raise ValueError('Expect 2D embeddings of shape (N, K)')
        regr = LinearRegression()
        embs = embs.cpu()
        labels = labels.cpu()
        regr.fit(embs, labels)
        preds = torch.tensor(regr.predict(embs))

        metrics = ['R2',    'concordance',      'NRMSE',                        'mse']
        fcns = [r2_score, concordance_corrcoef, normalized_root_mean_squared_error, mean_squared_error]
        scores = {}
        for metric, fcn in zip(metrics, fcns):
            scores[metric] = fcn(preds, labels)
        
        return scores