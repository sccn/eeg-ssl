from braindecode.datasets import BaseDataset, BaseConcatDataset
from braindecode.samplers import RecordingSampler, RelativePositioningSampler 
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from typing import List
import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.utils import check_random_state
from .ssl_utils import LitSSL

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
        super().__init__(self.info, **kwargs)

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
        DistributedSampler's iterator  contains indices of recordings specific to the current process
        """
        # XXX docstring missing
        return self.rng.choice(list(super().__iter__()))

    def sample_window(self, rec_ind=None):
        """Return a specific window.
        """
        # XXX docstring missing
        if rec_ind is None:
            rec_ind = self.sample_recording()
        win_ind = self.rng.choice(self.info.iloc[rec_ind]['index'])
        return win_ind, rec_ind

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
    """Sampler and Dataset object for relative positioning task from [Banville2020]_.

    Sample examples as tuples of two window indices, with a label indicating
    whether the windows are close or far, as defined by tau_pos and tau_neg.

    Parameters
    ----------
    metadata : pd.DataFrame
        See RecordingSampler.
    tau_pos_s : int
        Size of the positive context, in seconds. A positive pair contains two
        windows x1 and x2 which are separated by at most `tau_pos` samples.
    tau_neg_s : int
        Size of the negative context, in seconds. A negative pair contains two
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

    References
    ----------
    .. [Banville2020] Banville, H., Chehab, O., Hyvärinen, A., Engemann, D. A.,
           & Gramfort, A. (2020). Uncovering the structure of clinical EEG
           signals with self-supervised learning.
           arXiv preprint arXiv:2007.16104.
    """
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
            sampler = RelativePositioning.DistributedRelativePositioningSampler(dataset.get_metadata(), tau_pos, tau_neg, self.n_samples_per_dataset, self.tau_max, self.same_rec_neg, random_state=self.random_state)
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

    class DistributedRelativePositioningSampler(DistributedRecordingSampler):
        '''
        Note a difference in argument compared to non-distributed sampler:
        We provide n_samples_per_dataset so it can compute the number of examples
        for each subset of recordings accordingly
        '''
        def __init__(self, metadata, tau_pos, tau_neg, n_samples_per_dataset, 
                    tau_max=None, same_rec_neg=True, random_state=None, **kwargs):
            super().__init__(metadata, random_state=random_state, **kwargs)
            self.tau_pos = tau_pos
            self.tau_neg = tau_neg
            self.tau_max = np.inf if tau_max is None else tau_max
            self.same_rec_neg = same_rec_neg
            if not same_rec_neg and self.n_recordings < 2:
                raise ValueError('More than one recording must be available when '
                                'using across-recording negative sampling.')

            self.n_examples = n_samples_per_dataset * self.n_recordings
            print(f"Device {self.rank} - Number of datasets:", self.n_recordings)
            print(f"Device {self.rank} - Number of samples:", self.n_examples) 

        def _sample_pair(self):
            """Sample a pair of two windows.
            """
            # Sample first window
            win_ind1, rec_ind1 = self.sample_window()
            ts1 = self.metadata.iloc[win_ind1]['i_start_in_trial']
            ts = self.info.iloc[rec_ind1]['i_start_in_trial']

            # Decide whether the pair will be positive or negative
            pair_type = self.rng.binomial(1, 0.5)
            win_ind2 = None
            if pair_type == 0:  # Negative example
                if self.same_rec_neg:
                    mask = (
                        ((ts <= ts1 - self.tau_neg) & (ts >= ts1 - self.tau_max)) |
                        ((ts >= ts1 + self.tau_neg) & (ts <= ts1 + self.tau_max))
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
                win_ind2 = self.rng.choice(self.info.iloc[rec_ind1]['index'][mask])

            return win_ind1, win_ind2, float(pair_type)

        def presample(self):
            """Presample examples.

            Once presampled, the examples are the same from one epoch to another.
            """
            self.examples = [self._sample_pair() for _ in range(self.n_examples)]
            return self

        def __iter__(self):
            """Iterate over pairs.

            Yields
            ------
                (int): position of the first window in the dataset.
                (int): position of the second window in the dataset.
                (float): 0 for negative pair, 1 for positive pair.
            """
            for i in range(self.n_examples):
                if hasattr(self, 'examples'):
                    yield self.examples[i]
                else:
                    yield self._sample_pair()
                    
        def __len__(self):
            return self.n_examples
    
    class RelativePositioningLit(LitSSL):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.save_hyperparameters()
            self.clf = nn.Linear(self.emb_size, 1)
        
        def training_step(self, batch, batch_idx):
            # training_step defines the train loop.
            # it is independent of forward
            X, y = batch
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
    