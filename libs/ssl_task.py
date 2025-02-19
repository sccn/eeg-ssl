from braindecode.datasets import BaseDataset, BaseConcatDataset
from braindecode.samplers import RelativePositioningSampler, DistributedRelativePositioningSampler
import torch.distributed as dist
import numpy as np
from typing import List
import torch

class SSLTask():
    def __init__(self):
        pass
    
    def dataset(self, datasets: torch.utils.data.Dataset):
        return datasets
    
    def sampler(self, dataset: BaseConcatDataset):
        return torch.utils.data.sampler.Sampler(dataset)

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
        if dist.is_initialized():
            sampler = DistributedRelativePositioningSampler(dataset.get_metadata(), tau_pos, tau_neg, self.n_samples_per_dataset, self.tau_max, self.same_rec_neg, random_state=self.random_state)
        else:
            sampler = RelativePositioningSampler(dataset.get_metadata(), tau_pos, tau_neg, self.n_samples_per_dataset, self.tau_max, self.same_rec_neg, random_state=self.random_state)

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
