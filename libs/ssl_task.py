import warnings
from braindecode.datasets import BaseDataset, BaseConcatDataset
from braindecode.samplers import RecordingSampler, RelativePositioningSampler #, DistributedRecordingSampler, DistributedRelativePositioningSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import numpy as np
from typing import List
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
from .ssl_utils import LitSSL, instantiate_module
from .ssl_model import BENDRContextualizer, ConvEncoderBENDR, BENDRLSTM
from .evaluation import RankMe, Regressor, get_subjects_labels, get_subject_predictions
from torchmetrics.functional.classification import binary_accuracy
from torchmetrics.functional import f1_score
from lightning.pytorch.utilities import grad_norm
from torchmetrics.functional.regression import concordance_corrcoef, r2_score, normalized_root_mean_squared_error, mean_squared_error, mean_absolute_error
import braindecode
from typing import Any, Optional, Union
class DistributedRecordingSampler(DistributedSampler):
    """Base sampler simplifying sampling from recordings in distributed setting.

    Parameters
    ----------
    metadata : pd.DataFrame
        DataFrame with at least one of {subject, session, run} columns for each
        window in the BaseConcatDataset to sample examples from. Normally
        obtained with `BaseConcatDataset.get_metadata()`. For instance,
        `metadata.head()` might look like this:

           i_window_in_trial  i_start_in_trial  i_stop_in_trial  target  subject    session    run
        0                  0                 0              500      -1        4  session_T  run_0
        1                  1               500             1000      -1        4  session_T  run_0
        2                  2              1000             1500      -1        4  session_T  run_0
        3                  3              1500             2000      -1        4  session_T  run_0
        4                  4              2000             2500      -1        4  session_T  run_0

    random_state : np.RandomState | int | None
        Random state.

    Attributes
    ----------
    info : pd.DataFrame
        Series with MultiIndex index which contains the subject, session, run
        and window indices information in an easily accessible structure for
        quick sampling of windows.
    n_recordings : int
        Number of recordings available.
    kwargs : dict
        Additional keyword arguments to pass to torch DistributedSampler.
        See https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
    """

    def __init__(
        self,
        metadata,
        random_state=None,
        **kwargs,
    ):
        self.metadata = metadata
        self.info = self._init_info(metadata)
        self.rng = check_random_state(random_state)
        # send information to DistributedSampler parent to handle data splitting among workers
        super().__init__(self.info, seed=random_state, **kwargs)

    def _init_info(self, metadata, required_keys=None):
        """Initialize ``info`` DataFrame.

        Parameters
        ----------
        required_keys : list(str) | None
            List of additional columns of the metadata DataFrame that we should
            groupby when creating ``info``.

        Returns
        -------
            See class attributes.
        """
        keys = [k for k in ["subject", "session", "run"] if k in self.metadata.columns]
        if not keys:
            raise ValueError(
                "metadata must contain at least one of the following columns: "
                "subject, session or run."
            )

        if required_keys is not None:
            missing_keys = [k for k in required_keys if k not in self.metadata.columns]
            if len(missing_keys) > 0:
                raise ValueError(f"Columns {missing_keys} were not found in metadata.")
            keys += required_keys

        metadata = metadata.reset_index().rename(columns={"index": "window_index"})
        info = (
            metadata.reset_index()
            .groupby(keys)[["index", "i_start_in_trial"]]
            .agg(["unique"])
        )
        info.columns = info.columns.get_level_values(0)

        return info

    def sample_recording(self):
        """Return a random recording index.
        super().__iter__() contains indices of datasets specific to the current process
        determined by the DistributedSampler
        """
        # XXX docstring missing
        return self.rng.choice(list(super().__iter__()))

    def sample_window(self, rec_ind=None):
        """Return a specific window."""
        # XXX docstring missing
        if rec_ind is None:
            rec_ind = self.sample_recording()
        win_ind = self.rng.choice(self.info.iloc[rec_ind]["index"])
        return win_ind, rec_ind

    @property
    def n_recordings(self):
        return super().__len__()
    
class DistributedRelativePositioningSampler(DistributedRecordingSampler):
    """Sample examples for the relative positioning task from [Banville2020]_ in distributed mode.

    Sample examples as tuples of two window indices, with a label indicating
    whether the windows are close or far, as defined by tau_pos and tau_neg.

    Parameters
    ----------
    metadata : pd.DataFrame
        See RecordingSampler.
    tau_pos : int
        Size of the positive context, in samples. A positive pair contains two
        windows x1 and x2 which are separated by at most `tau_pos` samples.
    tau_neg : int
        Size of the negative context, in samples. A negative pair contains two
        windows x1 and x2 which are separated by at least `tau_neg` samples and
        at most `tau_max` samples. Ignored if `same_rec_neg` is False.
    n_examples : int
        Number of pairs to extract.
    tau_max : int | None
        See `tau_neg`.
    same_rec_neg : bool
        If True, sample negative pairs from within the same recording. If
        False, sample negative pairs from two different recordings.
    random_state : None | np.RandomState | int
        Random state.
    kwargs: dict
        Additional keyword arguments to pass to torch DistributedSampler.
        See https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler

    References
    ----------
    .. [Banville2020] Banville, H., Chehab, O., Hyvärinen, A., Engemann, D. A.,
        & Gramfort, A. (2020). Uncovering the structure of clinical EEG
        signals with self-supervised learning.
        arXiv preprint arXiv:2007.16104.
    """

    def __init__(
        self,
        metadata,
        tau_pos,
        tau_neg,
        n_examples,
        tau_max=None,
        same_rec_neg=True,
        random_state=None,
        **kwargs,
    ):
        super().__init__(metadata, random_state=random_state, **kwargs)
        self.tau_pos = tau_pos
        self.tau_neg = tau_neg
        self.tau_max = np.inf if tau_max is None else tau_max
        self.same_rec_neg = same_rec_neg

        self.n_examples = n_examples // self.info.shape[0] * self.n_recordings
        warnings.warn(
            f"Rank {dist.get_rank()} - Number of datasets: {self.n_recordings}"
        )
        warnings.warn(f"Rank {dist.get_rank()} - Number of samples: {self.n_examples}")

        if not same_rec_neg and self.n_recordings < 2:
            raise ValueError(
                "More than one recording must be available when "
                "using across-recording negative sampling."
            )

    def _sample_pair(self):
        """Sample a pair of two windows."""
        # Sample first window
        win_ind1, rec_ind1 = self.sample_window()
        ts1 = self.metadata.iloc[win_ind1]["i_start_in_trial"]
        ts = self.info.iloc[rec_ind1]["i_start_in_trial"]

        # Decide whether the pair will be positive or negative
        pair_type = self.rng.binomial(1, 0.5)
        win_ind2 = None
        if pair_type == 0:  # Negative example
            if self.same_rec_neg:
                mask = ((ts <= ts1 - self.tau_neg) & (ts >= ts1 - self.tau_max)) | (
                    (ts >= ts1 + self.tau_neg) & (ts <= ts1 + self.tau_max)
                )
            else:
                rec_ind2 = rec_ind1
                while rec_ind2 == rec_ind1:
                    win_ind2, rec_ind2 = self.sample_window()
        elif pair_type == 1:  # Positive example
            mask = (ts >= ts1 - self.tau_pos) & (ts <= ts1 + self.tau_pos)

        if win_ind2 is None:
            mask[ts == ts1] = False  # same window cannot be sampled twice
            if sum(mask) == 0:
                raise NotImplementedError
            win_ind2 = self.rng.choice(self.info.iloc[rec_ind1]["index"][mask])

        return win_ind1, win_ind2, float(pair_type)

    def presample(self):
        """Presample examples.

        Once presampled, the examples are the same from one epoch to another.
        """
        self.examples = [self._sample_pair() for _ in range(self.n_examples)]
        return self

    def __iter__(self):
        """
        Iterate over pairs.

        Yields
        ------
        int
            Position of the first window in the dataset.
        int
            Position of the second window in the dataset.
        float
            0 for a negative pair, 1 for a positive pair.
        """
        for i in range(self.n_examples):
            if hasattr(self, "examples"):
                yield self.examples[i]
            else:
                yield self._sample_pair()

    def __len__(self):
        return self.n_examples
    
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
        
        # def on_validation_epoch_end(self):
        #     pass
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
            '''
            NT-Xent (the normalized temperature-scaled cross entropy loss).
            '''
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

        # def on_validation_epoch_end(self):
        #     pass
    
class CPC(SSLTask):
    def __init__(self):
        super().__init__()
        # set all arguments except datasets as attributes

    def dataset(self, datasets: List[BaseConcatDataset]):
        return BaseConcatDataset(datasets)
    
    def sampler(self, dataset: BaseConcatDataset):
        return None

    class CPCLit(LitSSL):
        # Repurpose from https://github.com/SPOClab-ca/BENDR/blob/ac918abaec111d15fcaa2a8fcd2bd3d8b0d81a10/dn3_ext.py#L232
        def __init__(self, 
                    contextualizer_path: str,
                    contextualizer_kwargs: Optional[Union[dict[str, Any], dict[str, dict[str, Any]]]] = None, 
                    downsampling_factor=96, 
                    mask_rate=0.1, mask_span=6, temp=0.1,
                    permuted_encodings=False, permuted_contexts=False, enc_feat_l2=0.001,
                    unmasked_negative_frac=0.25, num_negatives=20, negative_same_recording=False, **kwargs):
            super().__init__(**kwargs)
            # self.contextualizer = BENDRContextualizer(
            #     in_features=self.encoder_emb_size,
            #     start_token=None,
            # )
            self.contextualizer = instantiate_module(contextualizer_path, contextualizer_kwargs)
            # Initialize replacement vector with standard normal
            self.mask_replacement = torch.nn.Parameter(torch.normal(0, self.encoder_emb_size**(-0.5), size=(self.encoder_emb_size,)),
                                                    requires_grad=True)

            self.predict_length = mask_span
            self._enc_downsample = downsampling_factor
            self.mask_rate = mask_rate
            self.mask_span = mask_span
            self.temp = temp
            self.permuted_encodings = permuted_encodings
            self.permuted_contexts = permuted_contexts
            self.beta = enc_feat_l2
            self.start_token = getattr(self.contextualizer, 'start_token', None)
            self.unmasked_negative_frac = unmasked_negative_frac
            self.num_negatives = num_negatives
            self.negative_same_recording = negative_same_recording

            self.evaluators = [Regressor(projection_head=True)]
        
        def _generate_negatives(self, z):
            """Generate negative samples to compare each sequence location against"""
            batch_size, feat, full_len = z.shape
            with torch.no_grad():
                z_k = z.clone().permute([0, 2, 1]).reshape(-1, feat)
                negative_inds = torch.randint(0, full_len-1, size=(batch_size, full_len * self.num_negatives))
                # From wav2vec 2.0 implementation, I don't understand
                # negative_inds[negative_inds >= candidates] += 1

                for i in range(1, batch_size):
                    negative_inds[i] += i * full_len

                z_k = z_k[negative_inds.view(-1)].view(batch_size, full_len, self.num_negatives, feat)
                return z_k, negative_inds

        def compute_contrastive_loss(self, true_z, c, mask_t=None):
            B, feat, seq_len = true_z.shape
            assert c.shape == true_z.shape, f"c {c.shape} and true_z {true_z.shape} should be the same shape"
            true_z = true_z.permute([0, 2, 1]) # (B, seq_len, F)
            c = c.permute([0, 2, 1]) # (B, seq_len, F)

            positives = F.cosine_similarity(c, true_z, dim=-1) / self.temp # (B, seq_len)
            if self.negative_same_recording and mask_t is not None:
                # gather False indices in mask_t for each batch item
                non_masked_inds = [torch.nonzero(~item)[:, 0] for item in mask_t]
                # for each batch item, sample with replacement self.num_negatives samples from negative_inds
                negatives_inds = []
                for i in range(B):
                    negatives_inds.append(non_masked_inds[i][torch.randint(0, non_masked_inds[i].shape[0], (self.num_negatives,))])
                negatives_inds = torch.stack(negatives_inds, dim=0).to(true_z.device) # (B, num_negatives)
                assert negatives_inds.shape == (B, self.num_negatives), f"negatives indices {negatives_inds.shape} should be (B, num_negatives)"
                assert ~torch.any(torch.gather(mask_t.to(true_z.device), dim=1, index=negatives_inds)), f"mask[negative_inds] should be False"
                
                negatives_batch = torch.gather(true_z, dim=1, index=negatives_inds.unsqueeze(-1).expand(-1, -1, feat)) # (B, num_negatives, F)
                negatives_batch = negatives_batch.unsqueeze(1) # (B, 1, num_negatives, F)
                assert negatives_batch.shape == (B, 1, self.num_negatives, feat), f"negatives {negatives_batch.shape} should be (B, num_negatives, F)"
                new_size = (B, seq_len, self.num_negatives, feat)
                negatives_batch = negatives_batch.expand(new_size)
            else:
                # create negative batch by randomize batch elements
                negatives_batch_ind = torch.randint(0, B, (B,), device=true_z.device)
                negatives_batch = true_z[negatives_batch_ind] # (B, seq_len, F)
                assert negatives_batch.shape == true_z.shape, f"negatives_batch {negatives_batch.shape} should be the same shape as true_z {true_z.shape}"
                negatives_inds = torch.randint(0, seq_len, (B, seq_len, self.num_negatives), device=true_z.device)
                negatives_batch = torch.gather(negatives_batch.unsqueeze(2).expand(-1, -1, self.num_negatives, -1),  # Expand to (B, seq_len, num_negative, F)
                                        dim=1,  # Sample along the time dimension (originally dimension 1)
                                        index=negatives_inds.unsqueeze(-1).expand(-1, -1, -1, feat))

            # negatives_batch, negatives_seq_ind = self._generate_negatives_from_batch(true_z)
            negatives = F.cosine_similarity(c.unsqueeze(-2), negatives_batch, dim=-1) / self.temp # (B, seq_len, num_negatives)

            # assert positives.shape == negatives.shape, f"positives {positives.shape} and negatives {negatives.shape} should be the same shape"
            # assert not torch.allclose(positives, negatives), f"positives {positives.shape} and negatives {negatives.shape} should not be the same"

            # negatives = negatives.unsqueeze(-1).expand(-1, -1, self.num_negatives)
            assert negatives.shape == (B, seq_len, self.num_negatives), f"negatives {negatives.shape} should be (B, seq_len, num_negatives)"

            positives = positives.unsqueeze(-1) # (B, seq_len, 1)
            assert positives.shape == (B, seq_len, 1), f"positives {positives.shape} should be (B, seq_len, 1)"

            logits = torch.cat([positives, negatives], dim=-1) # (B, seq_len, 1+num_negatives)
            logits = logits * 10

            logits = logits.view(-1, logits.shape[-1]) # flatten B x seq_len. Last dim correspond to torch CrossEntropyLoss C 
                                                        # --> will have true class label 0
            
            labels = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.long)
            # labels - (B x seq_len) all 0s
            # print('labels', labels)

            loss = torch.nn.functional.cross_entropy(logits, labels)

            return loss

        # def calculate_loss(self, inputs, outputs):
        #     logits = outputs[0]
        #     labels = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.long)
        #     # Note the loss_fn here integrates the softmax as per the normal classification pipeline (leveraging logsumexp)
        #     return torch.nn.functional.cross_entropy(logits, labels) + self.beta * outputs[1].pow(2).mean()

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
            # mask - (B, seq_len)
            return mask

        def generate_negatives_from_batch(self, z):
            """Generate negatives from other samples in the batch"""
            batch_size, feat, full_len = z.shape
            with torch.no_grad():
                z_k = z.clone().permute([0, 2, 1]).reshape(-1, feat)
                negative_inds = torch.randint(0, full_len-1, size=(batch_size, full_len * self.num_negatives))
                # From wav2vec 2.0 implementation, I don't understand
                # negative_inds[negative_inds >= candidates] += 1

                for i in range(1, batch_size):
                    negative_inds[i] += i * full_len

                z_k = z_k[negative_inds.view(-1)].view(batch_size, full_len, self.num_negatives, feat)
                return z_k, negative_inds

        def training_step(self, batch, batch_idx):
            z = self.encoder(batch[0])
            batch_size, feat, samples = z.shape
            # z - (B, F, seq_len)

            unmasked_z = z.clone()
            
            mask = self._make_mask((batch_size, samples), self.mask_rate, samples, self.mask_span)
            # make simple mask: only predict the last token
            # mask = torch.zeros((batch_size, samples), dtype=torch.bool)
            # mask[:, -1] = True

            z.transpose(2, 1)[mask] = self.mask_replacement

            c = self.contextualizer(z)
            # c - (B, F, seq_len) 

            loss = self.compute_contrastive_loss(unmasked_z, c, mask)

            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

            return loss
        
        def validation_step(self, batch, batch_idx):
            z = self.encoder(batch[0])
            batch_size, feat, samples = z.shape
            Y, subjects = batch[1], batch[3]
            # z - (B, F, seq_len)

            unmasked_z = z.clone()
            
            mask = None
            # mask = self._make_mask((batch_size, samples), self.mask_rate, samples, self.mask_span)
            # make simple mask: only predict the last token
            # mask = torch.zeros((batch_size, samples), dtype=torch.bool)
            # mask[:, -1] = True

            if mask is not None:
                z.transpose(2, 1)[mask] = self.mask_replacement

            c = self.contextualizer(z)
            
            loss = self.compute_contrastive_loss(unmasked_z, c, mask)

            self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

            for evaluator in self.evaluators:
                c_last = c[:, :, -1]
                evaluator((c_last, Y, subjects))
        
        def test_step(self, batch, batch_idx):
            z = self.encoder(batch[0])
            batch_size, feat, samples = z.shape
            Y, subjects = batch[1], batch[3]
            # z - (B, F, seq_len)

            unmasked_z = z.clone()
            
            c = self.contextualizer(z)
            
            loss = self.compute_contrastive_loss(unmasked_z, c)

            self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

            for evaluator in self.evaluators:
                c_last = c[:, :, -1]
                evaluator((c_last, Y, subjects))

            
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
            if isinstance(self.encoder, braindecode.models.deep4.Deep4Net):
                print('set bias')
                with torch.no_grad():
                    self.encoder.final_layer.conv_classifier.bias.copy_(torch.tensor(0.04))

            self.evaluators = [Regressor()]
            
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
            for evaluator in self.evaluators:
                evaluator.update((Z, Y, subjects))
        
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
            
    def sampler(self, dataset: BaseConcatDataset):
        return None
    def dataset(self, datasets: List[BaseConcatDataset]):
        return BaseConcatDataset(datasets)

    class ClassificationLit(LitSSL):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def normalize_data(self, x):
            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True) + 1e-7  # add small epsilon for numerical stability
            x = (x - mean) / std
            return x

        def training_step(self, batch, batch_idx):
            # self.train()
            # training_step defines the train loop.
            # it is independent of forward
            X, Y = batch[0], batch[1]
            Z = self.encoder(self.normalize_data(X))
            predictions = torch.argmax(Z, dim=1) 
            loss = nn.functional.cross_entropy(Z, Y)
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log('train_accuracy', binary_accuracy(predictions, Y), on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log('train_f1', f1_score(predictions, Y, task='binary'), on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return loss
        
        def validation_step(self, batch, batch_idx):
            X, Y, _, subjects = batch
            Z = self.encoder(self.normalize_data(X))
            predictions = torch.argmax(Z, 1) 
            loss = nn.functional.cross_entropy(Z, Y)
            self.log("val_Classifier/loss", loss, on_epoch=True, prog_bar=True, logger=True)
            self.log('val_Classifier/accuracy', binary_accuracy(predictions, Y), on_epoch=True, prog_bar=True, logger=True)
            self.log('val_Classifier/f1', f1_score(predictions, Y, task='binary'), on_epoch=True, prog_bar=True, logger=True)

        def on_validation_epoch_end(self):
            pass
