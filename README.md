# eeg-ssl
Apply Self-supervised Learning on NEMAR data

# Original video from Grandfort

https://www.youtube.com/watch?v=gm3a7T2bmnc

# Papers
Y. Benchetrit, H. Banville, and J.-R. King, “Brain decoding: toward real-time reconstruction of visual perception,” Mar. 14, 2024, arXiv: arXiv:2310.19812. doi: 10.48550/arXiv.2310.19812.

A. Thual et al., “Aligning brain functions boosts the decoding of visual semantics in novel subjects,” Dec. 11, 2023, arXiv: arXiv:2312.06467. doi: 10.48550/arXiv.2312.06467.

Banville, Hubert & Chehab, Omar & Hyvarinen, Aapo & Engemann, Denis-Alexander & Gramfort, Alexandre. (2020). Uncovering the structure of clinical EEG signals with self-supervised learning. Journal of Neural Engineering. 18. 10.1088/1741-2552/abca18. 
https://www.researchgate.net/publication/346857471_Uncovering_the_structure_of_clinical_EEG_signals_with_self-supervised_learning

Hubert Banville, Isabela Albuquerque, Aapo Hyvärinen, Graeme Moffat, Denis-Alexander Engemann, et al.. Self-supervised representation learning from electroencephalography signals. MLSP 2019 - IEEE 29th International Workshop on Machine Learning for Signal Processing, Oct 2019, Pittsburgh, United States. ⟨hal-02361350⟩
https://hal.science/hal-02361350

## Installation
- Install PyTorch for your respective system: https://pytorch.org/get-started/locally/
- Install braindecode: `pip install braindecode`

To run model training, also install:
- Install tensorboard: `pip install tensorboard`
- Install torch lightning: `pip install lightning`

## Run experiment using Docker
On a machine with Nvidia GPU:

```
docker run -it --runtime=nvidia --gpus all -v /mnt/nemar/child-mind-rest:/mnt/nemar/child-mind-rest -v .:/app dtyoung/eeg-ssl python main.py --nsubjects=30
```

Here the path to the dataset is at `/mnt/nemar/child-mind-rest` and we assume that the command is run in the top-level directory of the cloned version of this repo.