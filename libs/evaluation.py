import sys
sys.path.insert(0, '/home/dung/eeg-ssl')
import os
import torch
import torch.nn as nn
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
        embs = data[0]
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
    def __init__(self, projection_head=True, **kwargs):
        super().__init__(**kwargs)
        self.add_state("x", default=[], dist_reduce_fx='cat')
        self.add_state("labels", default=[], dist_reduce_fx='cat')
        self.add_state("subjects", default=[], dist_reduce_fx="cat")
        self.projection_head = projection_head

    def update(self, data:tuple) -> None:
        x = data[0]
        labels = data[1]
        subjects = data[2]
        self.x.append(x)
        self.labels.append(labels)
        subjects_encoded = encode_subjects(subjects, device=x.device)
        self.subjects.append(subjects_encoded)

    def compute(self) -> torch.Tensor:
        x = dim_zero_cat(self.x).float()
        labels = dim_zero_cat(self.labels).float()
        subjects_encoded = dim_zero_cat(self.subjects).float()
        
        if not self.projection_head:
            print('Regressor: using projection head')
            from sklearn.neural_network import MLPRegressor
            # from sklearn.linear_model import LinearRegression
            regr = MLPRegressor(random_state=1, max_iter=100)
            # Define model
            x_clone = x.clone().cpu()
            regr.fit(x_clone, labels.cpu())
            preds = regr.predict(x_clone)
            preds = torch.from_numpy(preds).to(x.device)
        else:
            print('Regressor: not using projection head')
            preds = x
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
            subject_labels_predictions.append([subject, label, subject_predictions[subject]]) # guarantee the same subject for label and prediction
        if len(subject_labels_predictions) > 1:
            subject_labels = torch.tensor([label for _, label, _ in subject_labels_predictions])
            subject_predictions_with_mean = torch.tensor([pred['mean'] for _, _, pred in subject_labels_predictions])
            subject_predictions_with_median = torch.tensor([pred['median'] for _, _, pred in subject_labels_predictions])
            subject_predictions_iqr = torch.tensor([pred['IQR'] for _, _, pred in subject_labels_predictions])
            subject_predictions_std = torch.tensor([pred['std'] for _, _, pred in subject_labels_predictions])
            
            for metric, fcn in zip(metrics, fcns):
                scores[f"subject_with_mean_{metric}"] = fcn(subject_predictions_with_mean, subject_labels)
                scores[f"subject_with_median_{metric}"] = fcn(subject_predictions_with_median, subject_labels)
            scores['subject_iqr_mean'] = torch.mean(subject_predictions_iqr)
            scores['subject_iqr_median'] = torch.median(subject_predictions_iqr)
            scores['subject_iqr_std'] = torch.std(subject_predictions_iqr)
            scores['subject_iqr_iqr'] = torch.quantile(subject_predictions_iqr, 0.75) - torch.quantile(subject_predictions_iqr, 0.25)
            scores['subject_std_mean'] = torch.mean(subject_predictions_std)
            scores['subject_std_median'] = torch.median(subject_predictions_std)
            scores['subject_std_std'] = torch.std(subject_predictions_std)
            scores['subject_std_iqr'] = torch.quantile(subject_predictions_std, 0.75) - torch.quantile(subject_predictions_std, 0.25)

        self.reset()

        return scores

def encode_subjects(subjects, device):
    return torch.stack([torch.tensor([ord(ch) for ch in subj], device=device) for subj in subjects])

def decode_subjects(subjects):
    return [''.join((chr(int(subjects[s,n].item())) for n in range(subjects.shape[1]))) for s in range(subjects.shape[0])]
    
def get_subjects_labels(subjects, labels):
    assert len(subjects) == labels.shape[0]
    subject_labels = defaultdict(list)
    for i, subject in enumerate(subjects):
        subject_labels[subject].append(labels[i])
    
    # check that all labels are the same for each subject
    for subject, lbls in subject_labels.items():
        if len(torch.unique(torch.tensor(lbls))) > 1:
            print(subject)
            print(torch.unique(torch.tensor(lbls)))
            raise ValueError(f"Subject {subject} has different labels: {set(lbls)}")
        # unique label --> assign to subject
        subject_labels[subject] = lbls[0]
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


def train_projection_layer_for_eval(model: nn.Module, 
                                    train_dataloader: torch.utils.data.DataLoader, 
                                    val_dataloader: torch.utils.data.DataLoader,
                                    regressor: None,
                                    encoder_emb_size=512, n_outputs=1, loss_fn=torch.nn.functional.mse_loss):
    from tqdm import tqdm

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # verify weights of model have not changed
    for param in model.parameters():
        param.requires_grad = False
    model.to(device)
    model.eval()

    projection_layer = regressor
    print('Using regression model:', projection_layer)
    embeddings = []
    labels = []
    for batch in tqdm(train_dataloader):
        x, y = batch[0], batch[1]
        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            z = model.embed(x)
            embeddings.append(z.cpu().numpy())
            labels.append(y.cpu().numpy())
    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)
    projection_layer.fit(embeddings, labels)
    print('Train score', projection_layer.score(embeddings, labels))

    # projection_layer = nn.Sequential(
    #     nn.Linear(encoder_emb_size, 100),
    #     nn.ReLU(),
    #     nn.Linear(100, n_outputs)
    # )

    # # train projection layer
    # projection_layer.to(device)
    # projection_layer.train()
    # optimizer = torch.optim.Adam(projection_layer.parameters(), lr=1e-3)
    # print('Training projection layer...')
    # for e in range(10):
    #     epoch_loss = 0
    #     for batch in tqdm(train_dataloader):
    #         optimizer.zero_grad()

    #         x, y = batch[0], batch[1]
    #         x, y = x.to(device), y.to(device)
    #         x, y = x.float(), y.float()

    #         z = model.embed(x)

    #         preds = projection_layer(z).squeeze()
            
    #         loss = loss_fn(preds, y)
    #         epoch_loss += loss.item()

    #         loss.backward()
    #         optimizer.step()

    #     epoch_loss /= len(train_dataloader)
    #     print(f'Epoch {e} - Loss: {epoch_loss:.4f}')

    # evaluate on validation set
    # print('Evaluating on validation set...')
    # projection_layer.eval()
    preds = []
    labels = []
    for batch in tqdm(val_dataloader):
        x, y = batch[0], batch[1]
        x, y = x.to(device), y.to(device)

        with torch.no_grad():
            z = model.embed(x)
            # preds.append(projection_layer(z))
            preds.append(torch.from_numpy(projection_layer.predict(z.cpu().numpy())))
            labels.append(y)
    preds = torch.cat(preds, dim=0).squeeze().cpu()
    labels = torch.cat(labels, dim=0).squeeze().cpu()
    metrics = ['R2',    'concordance',      'mse',                'mae']
    fcns = [r2_score, concordance_corrcoef, mean_squared_error, mean_absolute_error]
    score = {}
    for metric, fcn in zip(metrics, fcns):
        score[metric] = fcn(preds, labels).item()

    # print dictionary key value pair one per line
    print('Validation scores:')
    for k, v in score.items():
        print(f'\t{k}: {v:.4f}')

    return score

