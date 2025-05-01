import sys
sys.path.insert(0, '/home/dung/eeg-ssl')
import os
import torch
from torchmetrics import Metric
from torchmetrics.functional.regression import concordance_corrcoef, r2_score, normalized_root_mean_squared_error, mean_squared_error, mean_absolute_error
from torchmetrics.utilities import dim_zero_cat
import numpy as np
from collections import defaultdict

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
        embs = data
        self.embs.append(embs)

    def compute(self) -> torch.Tensor:
        # parse inputs
        # print('RankMe self.embs', self.embs)
        embs = dim_zero_cat(self.embs).float()
        if len(embs.shape) > 2:
            raise ValueError('Expect 2D embeddings of shape (N, K)')
        print('RankMe embs shape', embs.shape)
        if embs.shape[0] < embs.shape[1]:
            raise ValueError(f'Expect N >= K but received ({embs.shape})')
        # subselect 25600 embeddings randomly
        # embs = embs[torch.randperm(embs.shape[0])[:25600]]
        _, S, _ = torch.linalg.svd(embs)
        eps = 1e-7
        p = S/torch.linalg.norm(S, ord=1) + eps
        rank_z = torch.exp(-torch.sum(p*torch.log(p)))

        return rank_z

class Regressor(Metric):
    '''
    Validation using regression on target label
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("predictions", default=[], dist_reduce_fx='cat')
        self.add_state("labels", default=[], dist_reduce_fx='cat')
        self.add_state("subjects", default=[], dist_reduce_fx="cat")
        # self.subjects = []

    def update(self, data:tuple) -> None:
        predictions = data[0]
        labels = data[1]
        subjects = data[2]
        self.predictions.append(predictions)
        self.labels.append(labels)
        subjects_encoded = encode_subjects(subjects).to(device=self.device)
        # print('encoded subjects', subjects_encoded)
        self.subjects.append(subjects_encoded)
        # self.subjects.extend(subjects)

    def compute(self) -> torch.Tensor:
        preds = dim_zero_cat(self.predictions).float()
        labels = dim_zero_cat(self.labels).float()
        subjects_encoded = dim_zero_cat(self.subjects).float()
        
        # compute sample-level metrics
        metrics = ['R2',    'concordance',      'NRMSE',                        'mse',                  'mae']
        fcns = [r2_score, concordance_corrcoef, normalized_root_mean_squared_error, mean_squared_error, mean_absolute_error]
        scores = {}
        for metric, fcn in zip(metrics, fcns):
            scores[metric] = fcn(preds, labels)

        subjects = decode_subjects(subjects_encoded) # decode 
        
        # compute subject-level metrics
        subject_labels = get_subjects_labels(subjects, labels)
        subject_predictions = get_subject_predictions(subjects, preds)
        subject_labels_predictions = []
        for subject, label in subject_labels.items():
            subject_labels_predictions.append((subject, label, subject_predictions[subject])) # guarantee the same subject for label and prediction
        if len(subject_labels_predictions) > 1:
            subject_labels = torch.from_numpy(np.array([label for _, label, _ in subject_labels_predictions]))
            subject_predictions_with_mean = torch.from_numpy(np.array([pred['mean'] for _, _, pred in subject_labels_predictions]))
            subject_predictions_with_median = torch.from_numpy(np.array([pred['median'] for _, _, pred in subject_labels_predictions]))
            subject_predictions_iqr = np.array([pred['IQR'] for _, _, pred in subject_labels_predictions])
            subject_predictions_std = np.array([pred['std'] for _, _, pred in subject_labels_predictions])
            
            for metric, fcn in zip(metrics, fcns):
                scores[f"subject_with_mean_{metric}"] = fcn(subject_predictions_with_mean, subject_labels)
                scores[f"subject_with_median_{metric}"] = fcn(subject_predictions_with_median, subject_labels)
            scores['subject_iqr_mean'] = np.mean(subject_predictions_iqr)
            scores['subject_iqr_median'] = np.median(subject_predictions_iqr)
            scores['subject_iqr_std'] = np.std(subject_predictions_iqr)
            scores['subject_iqr_iqr'] = np.quantile(subject_predictions_iqr, 0.75) - np.quantile(subject_predictions_iqr, 0.25)
            scores['subject_std_mean'] = np.mean(subject_predictions_std)
            scores['subject_std_median'] = np.median(subject_predictions_std)
            scores['subject_std_std'] = np.std(subject_predictions_std)
            scores['subject_std_iqr'] = np.quantile(subject_predictions_std, 0.75) - np.quantile(subject_predictions_std, 0.25)
        
        return scores

def encode_subjects(subjects):
    return torch.tensor([[ord(ch) for ch in subj] for subj in subjects])

def decode_subjects(subjects):
    return [''.join((chr(int(subjects[s,n].item())) for n in range(subjects.shape[1]))) for s in range(subjects.shape[0])]
    
def get_subjects_labels(subjects, labels):
    assert len(subjects) == labels.shape[0]
    subject_labels = defaultdict(list)
    for i, subject in enumerate(subjects):
        subject_labels[subject].append(labels[i].cpu().numpy())
    
    # check that all labels are the same for each subject
    for subject, labels in subject_labels.items():
        if len(np.unique(labels)) > 1:
            raise ValueError(f"Subject {subject} has different labels: {set(labels)}")
        subject_labels[subject] = labels[0]
    return subject_labels

def get_subject_predictions(subjects, sample_predictions):
    assert len(sample_predictions) == len(subjects)
    subject_predictions = defaultdict(list)
    for i, subject in enumerate(subjects):
        subject_predictions[subject].append(sample_predictions[i].cpu().numpy())
    
    # for each subject, compute the mean and std of the predictions
    for subject, predictions in subject_predictions.items():
        subject_predictions[subject] = {
            'mean': np.mean(predictions),
            'std': np.std(predictions),
            'median': np.median(predictions),
            'IQR': np.quantile(predictions, 0.75) - np.quantile(predictions, 0.25)
        }
    return subject_predictions

