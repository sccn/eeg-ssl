from braindecode.datasets import BaseDataset, BaseConcatDataset
from braindecode.samplers import RelativePositioningSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from sklearn.utils import check_random_state
from typing import Optional
import numpy as np
from typing import List

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
    num_replicas (int, optional): Number of processes participating in
        distributed training. By default, :attr:`world_size` is retrieved from the
        current distributed group.
    rank (int, optional): Rank of the current process within :attr:`num_replicas`.
        By default, :attr:`rank` is retrieved from the current distributed
        group.
    shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
        indices.
    drop_last (bool, optional): if ``True``, then the sampler will drop the
        tail of the data to make it evenly divisible across the number of
        replicas. If ``False``, the sampler will add extra indices to make
        the data evenly divisible across the replicas. Default: ``False``.

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
            num_replicas: Optional[int] = None,
            rank: Optional[int] = None,
            shuffle: bool = True,
            drop_last: bool = False
    ):
        self.metadata = metadata
        self.info = self._init_info(metadata)
        self.rng = check_random_state(random_state)
        if not dist.is_available():
            raise RuntimeError("Requires distributed package to be available")
        rank = dist.get_rank()
        # send information to DistributedSampler parent to handle data splitting among workers
        super().__init__(self.info, num_replicas, rank, shuffle, random_state, drop_last)
        # super iter should contain only indices of datasets specific to the current process
        self._iterator = list(super().__iter__())

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
        keys = [k for k in ['subject', 'session', 'run']
                if k in self.metadata.columns]
        if not keys:
            raise ValueError(
                'metadata must contain at least one of the following columns: '
                'subject, session or run.')

        if required_keys is not None:
            missing_keys = [
                k for k in required_keys if k not in self.metadata.columns]
            if len(missing_keys) > 0:
                raise ValueError(
                    f'Columns {missing_keys} were not found in metadata.')
            keys += required_keys

        metadata = metadata.reset_index().rename(
            columns={'index': 'window_index'})
        info = metadata.reset_index().groupby(keys)[
            ['index', 'i_start_in_trial']].agg(['unique'])
        info.columns = info.columns.get_level_values(0)

        return info

    def sample_recording(self):
        """Return a random recording index.
        self._iterator contains indices of datasets specific to the current process
        determined by the DistributedSampler
        """
        # XXX docstring missing
        return self.rng.choice(self._iterator)

    def sample_window(self, rec_ind=None):
        """Return a specific window.
        """
        # XXX docstring missing
        if rec_ind is None:
            rec_ind = self.sample_recording()
        win_ind = self.rng.choice(self.info.iloc[rec_ind]['index'])
        return win_ind, rec_ind

    @property
    def n_recordings(self):
        return self.info.shape[0]

class SSLTask():
    def __init__(self):
        pass
    
class RelativePositioning(SSLTask):
    """Sampler and Dataset object for relative positioning task from [Banville2020]_.

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

    References
    ----------
    .. [Banville2020] Banville, H., Chehab, O., Hyvärinen, A., Engemann, D. A.,
           & Gramfort, A. (2020). Uncovering the structure of clinical EEG
           signals with self-supervised learning.
           arXiv preprint arXiv:2007.16104.
    """
    def __init__(self, datasets: List[BaseConcatDataset], tau_pos, tau_neg, n_samples_per_dataset, 
                 tau_max=None, same_rec_neg=True, random_state=None):
        if not dist.is_initialized():
            sampler = DistributedRelativePositioningSampler(datasets, tau_pos, tau_neg, n_samples_per_dataset, tau_max, same_rec_neg, random_state=random_state)
        else:
            sampler = RelativePositioningSampler(tau_pos, tau_neg, n_samples_per_dataset, tau_max, same_rec_neg)
        
        self.sampler = sampler
        self.dataset = RelativePositioningDataset(datasets)

        
class DistributedRelativePositioningSampler(DistributedSampler):
    """Sample examples for the relative positioning task from [Banville2020]_.

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

    References
    ----------
    .. [Banville2020] Banville, H., Chehab, O., Hyvärinen, A., Engemann, D. A.,
           & Gramfort, A. (2020). Uncovering the structure of clinical EEG
           signals with self-supervised learning.
           arXiv preprint arXiv:2007.16104.
    """
    def __init__(self, datasets: List[BaseConcatDataset], tau_pos, tau_neg, n_samples_per_dataset, 
                 tau_max=None, same_rec_neg=True, random_state=None):
        super().__init__(datasets, random_state=random_state)
        self.tau_pos = tau_pos
        self.tau_neg = tau_neg
        self.tau_max = np.inf if tau_max is None else tau_max
        self.same_rec_neg = same_rec_neg
        if not same_rec_neg and self.n_recordings < 2:
            raise ValueError('More than one recording must be available when '
                             'using across-recording negative sampling.')

        self.n_examples = n_samples_per_dataset * len(self._iterator)
        print(f"rank {dist.get_rank()} - Number of datasets:", len(self._iterator))
        print(f"rank {dist.get_rank()} - Number of samples:", self.n_examples) 

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


