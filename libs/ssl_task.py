from braindecode.datasets import BaseDataset, BaseConcatDataset
from braindecode.samplers import RecordingSampler, RelativePositioningSampler, DistributedRecordingSampler, DistributedRelativePositioningSampler
import torch.distributed as dist
import numpy as np
from typing import List
import torch
import torch.nn.functional as F
import torch.nn as nn
from .ssl_utils import LitSSL
from .evaluation import Regressor, get_subjects_labels, get_subject_predictions
from torchmetrics.functional.classification import binary_accuracy
from torchmetrics.functional import f1_score
from lightning.pytorch.utilities import grad_norm
from torchmetrics.functional.regression import concordance_corrcoef, r2_score, normalized_root_mean_squared_error, mean_squared_error, mean_absolute_error

class SSLTask():
    def __init__(self):
        pass
    
    def dataset(self, datasets: torch.utils.data.Dataset):
        return datasets
    
    def sampler(self, dataset: BaseConcatDataset):
        return torch.utils.data.sampler.Sampler(dataset)
    
    def loss(self):
        pass

class RelativePositioning(SSLTask):
    def __init__(self, tau_pos_s, tau_neg_s, n_samples_per_dataset, 
                 tau_max=None, same_rec_neg=True, random_state=None):
        super().__init__()
        # set all arguments except datasets as attributes
        self.tau_pos_s = tau_pos_s
        self.tau_neg_s = tau_neg_s if tau_neg_s else 2 * tau_pos_s
        self.n_samples_per_dataset = n_samples_per_dataset
        self.tau_max = tau_max
        self.same_rec_neg = same_rec_neg
        self.random_state = random_state
        
    def dataset(self, datasets: List[BaseConcatDataset]):
        return RelativePositioning.RelativePositioningDataset(datasets)
    
    def sampler(self, dataset: BaseConcatDataset):
        sfreq = dataset.datasets[0].raw.info['sfreq']
        tau_pos = int(sfreq * self.tau_pos_s)
        tau_neg = int(sfreq * self.tau_neg_s)
        n_examples = self.n_samples_per_dataset * len(dataset.datasets)
        if dist.is_initialized():
            sampler = DistributedRelativePositioningSampler(dataset.get_metadata(), tau_pos, tau_neg, self.n_samples_per_dataset, self.tau_max, self.same_rec_neg, random_state=self.random_state)
        else:
            sampler = RelativePositioningSampler(dataset.get_metadata(), tau_pos, tau_neg, n_examples, self.tau_max, self.same_rec_neg, random_state=self.random_state)

        return sampler

    class RelativePositioningDataset(BaseConcatDataset):
        """BaseConcatDataset with __getitem__ that expects 2 indices and a target.
        """
        def __init__(self, list_of_ds):
            super().__init__(list_of_ds)
            self.return_pair = True

        def __getitem__(self, index):
            if self.return_pair:
                ind1, ind2, y = index
                return (super().__getitem__(ind1)[0],
                        super().__getitem__(ind2)[0]), y
            else:
                return super().__getitem__(index)

        @property
        def return_pair(self):
            return self._return_pair

        @return_pair.setter
        def return_pair(self, value):
            self._return_pair = value

    class RelativePositioningLit(LitSSL):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.save_hyperparameters()
            self.clf = nn.Linear(self.emb_size, 1)
        
        def training_step(self, batch, batch_idx):
            # training_step defines the train loop.
            # it is independent of forward
            X, y = batch[0], batch[1]
            x1, x2 = X[0], X[1]
            z1, z2 = self.embed(x1), self.embed(x2)
            z = torch.abs(z1 - z2)
            loss = nn.functional.binary_cross_entropy_with_logits(self.clf(z).flatten(), y)

            self.log("train_loss", loss)
            return loss

class VICReg(SSLTask):
    def __init__(self, 
                 tau_pos_s,  # window length from which to sample different views (segments) of the same recording
                 n_samples_per_dataset,
                 random_state=None,
    ):
        super().__init__()
        self.tau_pos_s = tau_pos_s
        self.n_samples_per_dataset = n_samples_per_dataset
        self.random_state = random_state
    
    class VICRegDataset(BaseConcatDataset):
        def __init__(self, list_of_ds):
            super().__init__(list_of_ds)
            self.return_pair = True

        def __getitem__(self, index):
            if self.return_pair:
                ind1, ind2 = index
                return (super().__getitem__(ind1)[0],
                        super().__getitem__(ind2)[0])
            else:
                return super().__getitem__(index)

        @property
        def return_pair(self):
            return self._return_pair

        @return_pair.setter
        def return_pair(self, value):
            self._return_pair = value

    class VICRegSampler(RecordingSampler):
        def __init__(self,
                    metadata,
                    tau_pos,
                    n_samples_per_dataset,
                    random_state=None,
            ):
            super().__init__(metadata, random_state)
            self.tau_pos = tau_pos
            self.n_examples = n_samples_per_dataset * self.n_recordings
            self.return_pair = True
        
        def _sample_pair(self):
            """Sample a positive pair of two windows."""
            # Sample first window
            win_ind1, rec_ind1 = self.sample_window()
            ts1 = self.metadata.iloc[win_ind1]["i_start_in_trial"]
            ts = self.info.iloc[rec_ind1]["i_start_in_trial"]

            # Positive example
            mask = (ts >= ts1 - self.tau_pos) & (ts <= ts1 + self.tau_pos)

            mask[ts == ts1] = False  # same window cannot be sampled twice
            if sum(mask) == 0:
                raise NotImplementedError
            win_ind2 = self.rng.choice(self.info.iloc[rec_ind1]["index"][mask])

            return win_ind1, win_ind2

        def __iter__(self):
            """Iterate over pairs.

            Yields
            ------
                (int): position of the first window in the dataset.
                (int): position of the second window in the dataset.
                (float): 0 for negative pair, 1 for positive pair.
            """
            for i in range(self.n_examples):
                if hasattr(self, "examples"):
                    yield self.examples[i]
                else:
                    yield self._sample_pair()

        def __len__(self):
            return self.n_examples
        
    class DistributedVICRegSampler(DistributedRecordingSampler):
        def __init__(self, 
                     metadata,
                     tau_pos,
                     n_samples_per_dataset,
                     random_state=None,
                     shuffle=None):
            super().__init__(metadata, random_state, shuffle)
            self.tau_pos = tau_pos
            self.n_examples = n_samples_per_dataset * self.n_recordings
            print(f"rank {dist.get_rank()} - Number of datasets:", self.n_recordings)
            print(f"rank {dist.get_rank()} - Number of examples:", self.n_examples)
        
        def _sample_pair(self):
            """Sample a positive pair of two windows."""
            # Sample first window
            win_ind1, rec_ind1 = self.sample_window()
            ts1 = self.metadata.iloc[win_ind1]["i_start_in_trial"]
            ts = self.info.iloc[rec_ind1]["i_start_in_trial"]

            # Positive example
            mask = (ts >= ts1 - self.tau_pos) & (ts <= ts1 + self.tau_pos)

            mask[ts == ts1] = False  # same window cannot be sampled twice
            if sum(mask) == 0:
                raise NotImplementedError
            win_ind2 = self.rng.choice(self.info.iloc[rec_ind1]["index"][mask])

            return win_ind1, win_ind2

        def __iter__(self):
            for i in range(self.n_examples):
                if hasattr(self, "examples"):
                    yield self.examples[i]
                else:
                    yield self._sample_pair()

        def __len__(self):
            return self.n_examples

    class VICRegLit(LitSSL):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.save_hyperparameters()
        
        def loss(self,z1, z2,  # [B, D]
                tau=1,
                math_lambda=25.0,
                mu=25.0,
                nu=1):
            n, d = z1.shape[0], z1.shape[1]
            # Bardes, A., Ponce, J. & LeCun, Y. VICReg. http://arxiv.org/abs/2105.04906 (2022).
            # variance
            def reg_std(x, eps=0.0001):
                return torch.sqrt(x.var(dim=0) + eps)        # equation (2)
            def var_loss(z):
                return torch.mean(F.relu(tau - reg_std(z))) 
            # covariance
            def off_diagonal(x):
                n, m = x.shape[0], x.shape[1]
                assert n == m
                return x.flatten()[:-1].view(n-1, n+1)[:,1:].flatten() #x * (1 - torch.eye(n))
            def covar_loss(z):
                y = z - z.mean(dim=0)
                C_z = (y.T @ y) / (n - 1) # equation (3)
                c_z = off_diagonal(C_z).pow_(2).sum() / d # equation (4)
                return c_z
            # invariance
            def invar_loss(z1, z2):
                return F.mse_loss(z1, z2)
            
            l = math_lambda*invar_loss(z1,z2) + mu*(var_loss(z1)+var_loss(z2)) + nu*(covar_loss(z1)+covar_loss(z2)) # equation (6)

            return l

        def training_step(self, batch, batch_idx):
            # training_step defines the train loop.
            # it is independent of forward
            X = batch
            x1, x2 = X[0], X[1]
            z1, z2 = self.embed(x1), self.embed(x2)
            loss = self.loss(z1, z2)

            self.log("train_loss", loss)
            return loss
            
    def dataset(self, datasets: List[BaseConcatDataset]):
        return VICReg.VICRegDataset(datasets)
    
    def sampler(self, dataset: BaseConcatDataset):
        sfreq = dataset.datasets[0].raw.info['sfreq']
        tau_pos = int(sfreq * self.tau_pos_s)
        if dist.is_initialized():
            sampler = VICReg.DistributedVICRegSampler(dataset.get_metadata(), tau_pos, self.n_samples_per_dataset, random_state=self.random_state)
        else:
            sampler = VICReg.VICRegSampler(dataset.get_metadata(), tau_pos, self.n_samples_per_dataset, random_state=self.random_state)

        return sampler
    
class Regression(SSLTask):
    """
    Simple Regression task to validate the pipeline
    """
    def __init__(self):
        super().__init__()
            
    class RegressionLit(LitSSL):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # self.linear_head = nn.Sequential(
            #     nn.LazyLinear(1024),
            #     nn.ELU(),
            #     nn.Dropout(dropout),
            #     nn.Linear(1024, 512),
            #     nn.ELU(),
            #     nn.Dropout(dropout),
            #     nn.Linear(512, 1),
            # )
            self.evaluator = Regressor()
        
        def training_step(self, batch, batch_idx):
            # training_step defines the train loop.
            # it is independent of forward
            X, Y = batch[0], batch[1]
            Y = Y.to(torch.float32)
            Z = torch.squeeze(self.encoder(X)) 
            loss = nn.functional.mse_loss(Z, Y.to(torch.float32))

            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

            metrics = ['R2',    'concordance',      'mse',                'mae']
            fcns = [r2_score, concordance_corrcoef, mean_squared_error, mean_absolute_error]
            for metric, fcn in zip(metrics, fcns):
                score = fcn(Z, Y)
                self.log(f'train_Regressor/{metric}', score, on_step=False, on_epoch=True, prog_bar=True, logger=True)

            return loss

        def validation_step(self, batch, batch_idx):
            X, Y, subjects = batch[0], batch[1], batch[3]
            Y = Y.to(torch.float32)
            Z = torch.squeeze(self.encoder(X))
            self.evaluator.update((Z, Y, subjects))
        
        def on_validation_epoch_end(self):
            scores = self.evaluator.compute()
            for k, v in scores.items():
                    self.log(f'val_Regressor/{k}', v, prog_bar=True, logger=True)

        def validation_step_not_metrics(self, batch, batch_idx):
            X, Y = batch[0], batch[1]
            Y = Y.to(torch.float32)
            Z = torch.squeeze(self.encoder(X))
            loss = nn.functional.mse_loss(Z, Y)
            self.log(f'val_Regressor/mse', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

            metrics = ['R2',    'concordance',      'NRMSE',                        'mse',                  'mae']
            fcns = [r2_score, concordance_corrcoef, normalized_root_mean_squared_error, mean_squared_error, mean_absolute_error]
            for metric, fcn in zip(metrics, fcns):
                score = fcn(Z, Y)
                self.log(f'val_Regressor/{metric}', score, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
    def dataset(self, datasets: List[BaseConcatDataset]):
        return BaseConcatDataset(datasets)
    
    def sampler(self, dataset: BaseConcatDataset):
        return None

class Classification(SSLTask):
    """
    Simple Regression task to validate the pipeline
    """
    def __init__(self):
        super().__init__()
            
    class ClassificationLit(LitSSL):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def on_before_optimizer_step(self, optimizer):
            # Compute the 2-norm for each layer
            # If using mixed precision, the gradients are already unscaled here
            norms = grad_norm(self.encoder, norm_type=2)
            self.log_dict(norms) 

        def training_step(self, batch, batch_idx):
            # self.train()
            # training_step defines the train loop.
            # it is independent of forward
            X, Y = batch[0], batch[1]
            Z = self.encoder(X)
            predictions = torch.nn.functional.softmax(Z, dim=1) 
            probs, predictions = torch.max(predictions, 1) # TODO assume that's the compatible way to cross entropy
            loss = nn.functional.cross_entropy(Z, Y)
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log('train_accuracy', binary_accuracy(predictions, Y), on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log('train_f1', f1_score(predictions, Y, task='binary'), on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return loss
        
        def validation_step(self, batch, batch_idx):
            X, Y, _, subjects = batch
            Z = self.encoder(X)
            predictions = torch.nn.functional.softmax(Z, dim=1) 
            probs, predictions = torch.max(predictions, 1) # TODO assume that's the compatible way to cross entropy
            loss = nn.functional.cross_entropy(Z, Y)
            self.log("val_Classifier/loss", loss, on_epoch=True, prog_bar=True, logger=True)
            self.log('val_Classifier/accuracy', binary_accuracy(predictions, Y), on_epoch=True, prog_bar=True, logger=True)
            self.log('val_Classifier/f1', f1_score(predictions, Y, task='binary'), on_epoch=True, prog_bar=True, logger=True)

        def on_validation_epoch_end(self):
            pass

    def dataset(self, datasets: List[BaseConcatDataset]):
        return BaseConcatDataset(datasets)
    
    def sampler(self, dataset: BaseConcatDataset):
        return None