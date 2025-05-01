from braindecode.datasets import BaseDataset, BaseConcatDataset
from braindecode.samplers import RecordingSampler, RelativePositioningSampler, DistributedRecordingSampler, DistributedRelativePositioningSampler
import torch.distributed as dist
import numpy as np
from typing import List
import torch
import torch.nn.functional as F
import torch.nn as nn
from .ssl_utils import LitSSL
from .evaluation import RankMe, Regressor, get_subjects_labels, get_subject_predictions
from torchmetrics.functional.classification import binary_accuracy
from torchmetrics.functional import f1_score
from lightning.pytorch.utilities import grad_norm
from torchmetrics.functional.regression import concordance_corrcoef, r2_score, normalized_root_mean_squared_error, mean_squared_error, mean_absolute_error
import braindecode

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
            print("Using non-distributed sampler")
            sampler = RelativePositioningSampler(dataset.get_metadata(), tau_pos, tau_neg, n_examples, self.tau_max, self.same_rec_neg)

        return sampler

    class RelativePositioningDataset(BaseConcatDataset):
        """BaseConcatDataset with __getitem__ that expects 2 indices and a target.
        """
        def __init__(self, list_of_ds):
            super().__init__(list_of_ds)
            self.return_pair = True

        def __getitem__(self, index):
            if self.return_pair:
                # print('index', index)
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
            self.projection_head = nn.Sequential(
                nn.Linear(self.emb_size, 1),
                nn.ReLU(),
            )
            self.evaluator = RankMe()
        
        def training_step(self, batch, batch_idx):
            # training_step defines the train loop.
            # it is independent of forward
            X, y = batch[0], batch[1]
            x1, x2 = X[0], X[1]
            z1, z2 = self.encoder(x1), self.encoder(x2)
            z = torch.abs(z1 - z2)
            # print('y', y)
            loss = nn.functional.binary_cross_entropy_with_logits(torch.squeeze(self.projection_head(z)), y)

            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return loss
        
        def validation_step(self, batch, batch_idx):
            pass
            # X, Y, subjects = batch[0], batch[1], batch[3]
            # Z = torch.squeeze(self.encoder(X))
            # self.evaluator.update(Z)
        
        def on_validation_epoch_end(self):
            pass
            # score = self.evaluator.compute()
            # self.log(f'val_RankMe', score, prog_bar=True, logger=True, sync_dist=True)
        
    

class SimCLR(SSLTask):
    def __init__(self, tau_pos_s):
        super().__init__()
        # set all arguments except datasets as attributes
        self.tau_pos_s = tau_pos_s
        
    class SimCLRSampler(RecordingSampler):
        def __init__(
            self,
            metadata,
            tau_pos,
            random_state=None,
        ):
            super().__init__(metadata, random_state=random_state)
            self.tau_pos = tau_pos

        def _sample_pair(self):
            """Sample a positive pair of two windows from same recording."""
            # Sample first window
            win_ind1, rec_ind1 = self.sample_window()
            ts1 = self.metadata.iloc[win_ind1]["i_start_in_trial"]
            ts = self.info.iloc[rec_ind1]["i_start_in_trial"]

            # Decide whether the pair will be positive or negative
            win_ind2 = None
            mask = (ts >= ts1 - self.tau_pos) & (ts <= ts1 + self.tau_pos)

            if win_ind2 is None:
                mask[ts == ts1] = False  # same window cannot be sampled twice
                if sum(mask) == 0:
                    raise NotImplementedError
                win_ind2 = self.rng.choice(self.info.iloc[rec_ind1]["index"][mask])

            return win_ind1, win_ind2

        def __iter__(self):
            """
            Iterate over pairs.

            Yields
            ------
            int
                Position of the first window in the dataset.
            int
                Position of the second window in the dataset.
            """
            for i in range(len(self.metadata)):
                yield self._sample_pair()

        def __len__(self):
            return len(self.metadata)
        
        @property
        def n_examples(self):
            return len(self.metadata)

    class SimCLRDataset(BaseConcatDataset):
        """BaseConcatDataset with __getitem__ that expects 2 indices and a target.
        """
        def __init__(self, list_of_ds):
            super().__init__(list_of_ds)

        def __getitem__(self, index):
            ind1, ind2 = index
            return (super().__getitem__(ind1)[0],
                    super().__getitem__(ind2)[0])

    def dataset(self, datasets: List[BaseConcatDataset]):
        return SimCLR.SimCLRDataset(datasets)
    
    def sampler(self, dataset: BaseConcatDataset):
        sfreq = dataset.datasets[0].raw.info['sfreq']
        tau_pos = int(sfreq * self.tau_pos_s)
        sampler = SimCLR.SimCLRSampler(dataset.get_metadata(), tau_pos)

        return sampler

    class SimCLRLit(LitSSL):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.save_hyperparameters()
            self.projection_head = nn.Sequential(
                nn.Linear(self.encoder_emb_size, self.emb_size),
                nn.ReLU(),
            )
            self.evaluator = RankMe()
        
        def contrastive_loss(self, hidden1, hidden2, temperature=0.1):
            hidden1, hidden2 = torch.nn.functional.normalize(hidden1, dim=1, p=2), torch.nn.functional.normalize(hidden2, dim=1, p=2)
            batch_size = hidden1.shape[0]
            masks = torch.nn.functional.one_hot(torch.arange(batch_size), batch_size).bool().to(hidden1.device)
            logits_aa = torch.matmul(hidden1, hidden1.T) / temperature
            logits_aa = logits_aa.masked_fill(masks == 1, float('-inf')) # self-embeddings should not contribute to negative loss
            logits_bb = torch.matmul(hidden2, hidden2.T) / temperature
            logits_bb = logits_bb.masked_fill(masks == 1, float('-inf')) # self-embeddings should not contribute to negative loss
            logits_ab = torch.matmul(hidden1, hidden2.T) / temperature
            logits_ba = torch.matmul(hidden2, hidden1.T) / temperature

            loss_a = torch.nn.functional.cross_entropy(torch.cat([logits_ab, logits_aa], dim=1), torch.arange(batch_size).to(hidden1.device))
            loss_b = torch.nn.functional.cross_entropy(torch.cat([logits_ba, logits_bb], dim=1), torch.arange(batch_size).to(hidden1.device))
            loss = (loss_a + loss_b) / 2

            return loss

        def training_step(self, batch, batch_idx):
            # training_step defines the train loop.
            # it is independent of forward
            temp = 0.1
            x1, x2 = batch[0], batch[1]
            h1, h2 = self.encoder(x1), self.encoder(x2)
            z1, z2 = self.projection_head(h1), self.projection_head(h2)
            assert z1.shape[0] == z2.shape[0] == x1.shape[0]
            
            loss = self.contrastive_loss(z1, z2, temperature=temp)
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return loss
        
        def validation_step(self, batch, batch_idx):
            temp = 0.1
            x1, x2 = batch[0], batch[1]
            h1, h2 = self.encoder(x1), self.encoder(x2)
            z1, z2 = self.projection_head(h1), self.projection_head(h2)
            assert z1.shape[0] == z2.shape[0] == x1.shape[0]
            
            loss = self.contrastive_loss(z1, z2, temperature=temp)
            self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return loss

        def on_validation_epoch_end(self):
            pass
    
class CPC(SSLTask):
    def __init__(self, tau_pos_s):
        super().__init__()
        # set all arguments except datasets as attributes
        self.tau_pos_s = tau_pos_s

    class BendingCollegeWav2Vec():
        """
        A more wav2vec 2.0 style of constrastive self-supervision, more inspired-by than exactly like it.
        """
        def __init__(self, encoder, context_fn, mask_rate=0.1, mask_span=6, learning_rate=0.01, temp=0.5,
                    permuted_encodings=False, permuted_contexts=False, enc_feat_l2=0.001, multi_gpu=False,
                    l2_weight_decay=1e-4, unmasked_negative_frac=0.25, encoder_grad_frac=1.0,
                    num_negatives=100, **kwargs):
            self.predict_length = mask_span
            self.best_metric = None
            self.mask_rate = mask_rate
            self.mask_span = mask_span
            self.temp = temp
            self.permuted_encodings = permuted_encodings
            self.permuted_contexts = permuted_contexts
            self.beta = enc_feat_l2
            self.start_token = getattr(context_fn, 'start_token', None)
            self.unmasked_negative_frac = unmasked_negative_frac
            self.num_negatives = num_negatives

        def _make_span_from_seeds(self, seeds, span, total=None):
            inds = list()
            for seed in seeds:
                for i in range(seed, seed + span):
                    if total is not None and i >= total:
                        break
                    elif i not in inds:
                        inds.append(int(i))
            return np.array(inds)

        def _make_mask(self, shape, p, total, span, allow_no_inds=False):
            # num_mask_spans = np.sum(np.random.rand(total) < p)
            # num_mask_spans = int(p * total)
            mask = torch.zeros(shape, requires_grad=False, dtype=torch.bool)

            for i in range(shape[0]):
                mask_seeds = list()
                while not allow_no_inds and len(mask_seeds) == 0 and p > 0:
                    mask_seeds = np.nonzero(np.random.rand(total) < p)[0]

                mask[i, self._make_span_from_seeds(mask_seeds, span, total=total)] = True

            return mask

        def description(self, sequence_len):
            encoded_samples = self._enc_downsample(sequence_len)
            desc = "{} samples | mask span of {} at a rate of {} => E[masked] ~= {}".format(
                encoded_samples, self.mask_span, self.mask_rate,
                int(encoded_samples * self.mask_rate * self.mask_span))
            return desc

        def _generate_negatives(self, z):
            """Generate negative samples to compare each sequence location against"""
            batch_size, feat, full_len = z.shape
            z_k = z.permute([0, 2, 1]).reshape(-1, feat)
            with torch.no_grad():
                # candidates = torch.arange(full_len).unsqueeze(-1).expand(-1, self.num_negatives).flatten()
                negative_inds = torch.randint(0, full_len-1, size=(batch_size, full_len * self.num_negatives))
                # From wav2vec 2.0 implementation, I don't understand
                # negative_inds[negative_inds >= candidates] += 1

                for i in range(1, batch_size):
                    negative_inds[i] += i * full_len

            z_k = z_k[negative_inds.view(-1)].view(batch_size, full_len, self.num_negatives, feat)
            return z_k, negative_inds

        def _calculate_similarity(self, z, c, negatives):
            c = c[..., 1:].permute([0, 2, 1]).unsqueeze(-2)
            z = z.permute([0, 2, 1]).unsqueeze(-2)

            # In case the contextualizer matches exactly, need to avoid divide by zero errors
            negative_in_target = (c == negatives).all(-1)
            targets = torch.cat([c, negatives], dim=-2)

            logits = F.cosine_similarity(z, targets, dim=-1) / self.temp
            if negative_in_target.any():
                logits[1:][negative_in_target] = float("-inf")

            return logits.view(-1, logits.shape[-1])

        def forward(self, *inputs):
            z = self.encoder(inputs[0])

            if self.permuted_encodings:
                z = z.permute([1, 2, 0])

            unmasked_z = z.clone()

            batch_size, feat, samples = z.shape

            if self._training:
                mask = self._make_mask((batch_size, samples), self.mask_rate, samples, self.mask_span)
            else:
                mask = torch.zeros((batch_size, samples), requires_grad=False, dtype=torch.bool)
                half_avg_num_seeds = max(1, int(samples * self.mask_rate * 0.5))
                if samples <= self.mask_span * half_avg_num_seeds:
                    raise ValueError("Masking the entire span, pointless.")
                mask[:, self._make_span_from_seeds((samples // half_avg_num_seeds) * np.arange(half_avg_num_seeds).astype(int),
                                                self.mask_span)] = True

            c = self.context_fn(z, mask)

            # Select negative candidates and generate labels for which are correct labels
            negatives, negative_inds = self._generate_negatives(z)

            # Prediction -> batch_size x predict_length x predict_length
            logits = self._calculate_similarity(unmasked_z, c, negatives)
            return logits, z, mask

        @staticmethod
        def _mask_pct(inputs, outputs):
            return outputs[2].float().mean().item()

        @staticmethod
        def _contrastive_accuracy(inputs, outputs):
            logits = outputs[0]
            labels = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.long)
            return StandardClassification._simple_accuracy([labels], logits)

        def calculate_loss(self, inputs, outputs):
            logits = outputs[0]
            labels = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.long)
            # Note the loss_fn here integrates the softmax as per the normal classification pipeline (leveraging logsumexp)
            return self.loss_fn(logits, labels) + self.beta * outputs[1].pow(2).mean()    

    class CPCSampler(RecordingSampler):
        def __init__(
            self,
            metadata,
            tau_pos,
            random_state=None,
        ):
            super().__init__(metadata, random_state=random_state)
            self.tau_pos = tau_pos

        def _sample_pair(self):
            """Sample a positive pair of two windows from same recording."""
            # Sample first window
            win_ind1, rec_ind1 = self.sample_window()
            ts1 = self.metadata.iloc[win_ind1]["i_start_in_trial"]
            ts = self.info.iloc[rec_ind1]["i_start_in_trial"]

            # Decide whether the pair will be positive or negative
            win_ind2 = None
            mask = (ts >= ts1 - self.tau_pos) & (ts <= ts1 + self.tau_pos)

            if win_ind2 is None:
                mask[ts == ts1] = False  # same window cannot be sampled twice
                if sum(mask) == 0:
                    raise NotImplementedError
                win_ind2 = self.rng.choice(self.info.iloc[rec_ind1]["index"][mask])

            return win_ind1, win_ind2

        def __iter__(self):
            """
            Iterate over pairs.

            Yields
            ------
            int
                Position of the first window in the dataset.
            int
                Position of the second window in the dataset.
            """
            for i in range(len(self.metadata)):
                yield self._sample_pair()

        def __len__(self):
            return len(self.metadata)
        
        @property
        def n_examples(self):
            return len(self.metadata)

    class CPCDataset(BaseConcatDataset):
        """BaseConcatDataset with __getitem__ that expects 2 indices and a target.
        """
        def __init__(self, list_of_ds):
            super().__init__(list_of_ds)

        def __getitem__(self, index):
            # print('index', index)
            ind1, ind2 = index
            return (super().__getitem__(ind1)[0],
                    super().__getitem__(ind2)[0])

    def dataset(self, datasets: List[BaseConcatDataset]):
        return SimCLR.SimCLRDataset(datasets)
    
    def sampler(self, dataset: BaseConcatDataset):
        sfreq = dataset.datasets[0].raw.info['sfreq']
        tau_pos = int(sfreq * self.tau_pos_s)
        sampler = SimCLR.SimCLRSampler(dataset.get_metadata(), tau_pos)

        return sampler

    class CPCLit(LitSSL):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.save_hyperparameters()
            self.projection_head = nn.Sequential(
                nn.Linear(self.encoder_emb_size, self.emb_size),
                nn.ReLU(),
            )
            self.evaluator = RankMe()
        
        def contrastive_loss(self, hidden1, hidden2, temperature=0.1, LARGE_NUM=1e9):
            hidden1, hidden2 = torch.nn.functional.normalize(hidden1, dim=1, p=2), torch.nn.functional.normalize(hidden2, dim=1, p=2)
            batch_size = hidden1.shape[0]
            labels = torch.nn.functional.one_hot(torch.arange(batch_size), batch_size * 2).to(hidden1.device)
            labels[:,batch_size:] = labels[:,:batch_size]
            labels = labels.float()  # Convert to float for loss calculation

            assert torch.allclose(torch.sum(labels, dim=1), torch.ones_like(torch.sum(labels, dim=1))*2)
            masks = torch.nn.functional.one_hot(torch.arange(batch_size), batch_size).bool().to(hidden1.device)
            logits_aa = torch.matmul(hidden1, hidden1.T) / temperature
            # logits_aa = logits_aa.masked_fill(masks == 1, float('-inf'))
            logits_bb = torch.matmul(hidden2, hidden2.T) / temperature
            # logits_bb = logits_bb.masked_fill(masks == 1, float('-inf'))
            logits_ab = torch.matmul(hidden1, hidden2.T) / temperature
            logits_ba = torch.matmul(hidden2, hidden1.T) / temperature
            loss_a = torch.nn.functional.cross_entropy(torch.cat([logits_ab, logits_aa], dim=1), labels)
            loss_b = torch.nn.functional.cross_entropy(torch.cat([logits_ba, logits_bb], dim=1), labels)
            loss = (loss_a + loss_b) / 2

            return loss

        def training_step(self, batch, batch_idx):
            # training_step defines the train loop.
            # it is independent of forward
            temp = 0.1
            x1, x2 = batch[0], batch[1]
            h1, h2 = self.encoder(x1), self.encoder(x2)
            z1, z2 = self.projection_head(h1), self.projection_head(h2)
            assert z1.shape[0] == z2.shape[0] == x1.shape[0]
            
            loss = self.contrastive_loss(z1, z2, temperature=temp)
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return loss
        
        def validation_step(self, batch, batch_idx):
            temp = 0.1
            x1, x2 = batch[0], batch[1]
            h1, h2 = self.encoder(x1), self.encoder(x2)
            z1, z2 = self.projection_head(h1), self.projection_head(h2)
            assert z1.shape[0] == z2.shape[0] == x1.shape[0]
            
            loss = self.contrastive_loss(z1, z2, temperature=temp)
            self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return loss

        def on_validation_epoch_end(self):
            pass
            
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
            if isinstance(self.encoder, braindecode.models.deep4.Deep4Net):
                print('set bias')
                with torch.no_grad():
                    self.encoder.final_layer.conv_classifier.bias.copy_(torch.tensor(0.04))

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
                self.log(f'val_Regressor/{k}', v, prog_bar=True, logger=True, sync_dist=True)

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