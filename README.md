# eeg-ssl
Apply Self-supervised Learning on NEMAR data

# Original video from Grandfort

https://www.youtube.com/watch?v=gm3a7T2bmnc

# Papers

Banville, Hubert & Chehab, Omar & Hyvarinen, Aapo & Engemann, Denis-Alexander & Gramfort, Alexandre. (2020). Uncovering the structure of clinical EEG signals with self-supervised learning. Journal of Neural Engineering. 18. 10.1088/1741-2552/abca18. 
https://www.researchgate.net/publication/346857471_Uncovering_the_structure_of_clinical_EEG_signals_with_self-supervised_learning

Hubert Banville, Isabela Albuquerque, Aapo Hyvärinen, Graeme Moffat, Denis-Alexander Engemann, et al.. Self-supervised representation learning from electroencephalography signals. MLSP 2019 - IEEE 29th International Workshop on Machine Learning for Signal Processing, Oct 2019, Pittsburgh, United States. ⟨hal-02361350⟩
https://hal.science/hal-02361350

## Run experiment using Docker
On a machine with Nvidia GPU:

```
docker run -it --runtime=nvidia --gpus all -v /mnt/nemar/child-mind-rest:/mnt/nemar/child-mind-rest -v .:/app eeg-ssl python main.py --nsubjects=30
```

Here the path to the dataset is at `/mnt/nemar/child-mind-rest` and we assume that the command is run in the top-level directory of the cloned version of this repo.
