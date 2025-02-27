# eeg-ssl
Apply Self-supervised Learning on NEMAR and EEGDASH data

## Installation
- Install our fork of mne-python for latest EEGLAB import behavior: `pip install git+https://github.com/dungscout96/mne-python.git`
- Install PyTorch for your respective system: https://pytorch.org/get-started/locally/
- Install braindecode: `pip install braindecode`

To run model training, also install:
- Install tensorboard: `pip install tensorboard`
- Install torch lightning: `pip install lightning`

## Run test model

Download example data
https://drive.google.com/file/d/1KWEDoZOqyLojq0hQx8lUNTWSdZ5tBlTc/view?usp=sharing

Place the zip file in ./notebooks/data

Run the notebook ./notebooks/HBN_braindecode.ipynb

## Original video from Grandfort

https://www.youtube.com/watch?v=gm3a7T2bmnc

## Papers
Y. Benchetrit, H. Banville, and J.-R. King, “Brain decoding: toward real-time reconstruction of visual perception,” Mar. 14, 2024, arXiv: arXiv:2310.19812. doi: 10.48550/arXiv.2310.19812.

A. Thual et al., “Aligning brain functions boosts the decoding of visual semantics in novel subjects,” Dec. 11, 2023, arXiv: arXiv:2312.06467. doi: 10.48550/arXiv.2312.06467.

Banville, Hubert & Chehab, Omar & Hyvarinen, Aapo & Engemann, Denis-Alexander & Gramfort, Alexandre. (2020). Uncovering the structure of clinical EEG signals with self-supervised learning. Journal of Neural Engineering. 18. 10.1088/1741-2552/abca18. 
https://www.researchgate.net/publication/346857471_Uncovering_the_structure_of_clinical_EEG_signals_with_self-supervised_learning

Hubert Banville, Isabela Albuquerque, Aapo Hyvärinen, Graeme Moffat, Denis-Alexander Engemann, et al.. Self-supervised representation learning from electroencephalography signals. MLSP 2019 - IEEE 29th International Workshop on Machine Learning for Signal Processing, Oct 2019, Pittsburgh, United States. ⟨hal-02361350⟩
https://hal.science/hal-02361350


## Run experiment using Docker
On a machine with Nvidia GPU:

```
docker run -v $(pwd):/app --rm -it --runtime=nvidia --gpus all --entrypoint bash dtyoung/eeg-ssl:linux
```

Here the path to the dataset is at `/mnt/nemar/child-mind-rest` and we assume that the command is run in the top-level directory of the cloned version of this repo.

## Voyager install

-  Download BIDS data and put it in the data folder. If you are downloading ds005505 for example, make sure the data is in data/ds005505/ (example of command scp -r arno@login.expanse.sdsc.edu:/expanse/projects/nemar/openneuro/ds005505 eeg-ssl/notebooks/data/)
  
-  Edit the file "runs/config_RP.yaml" and change the list of datasets under "data" to the data that is available to you (for example "ds005505") change also the "data_dir" entry to "data"

-  Edit the file "voyager.yaml" and change the "hostpath" to you home folder. Also uncomment the first metadata name "name: eeg-ssl-interactive" and comment the jupyter one. Uncomment the args parameter " [ "while true; do sleep 30; done;" ]" and comment the one that is active (could be multiple lines, for example 4 lines).

-  Start Kubernete

```
module load kubernetes/voyager/1.21.14
kubectl apply -f ./voyager.yaml
kubectl describe pod eeg-ssl-interactive
kubectl logs pod eeg-ssl-interactive
kubectl exec -it eeg-ssl-interactive  -- /bin/bash
```

- Once on the Kubernete, run the program

```
python main.py fit --config runs/config_RP.yaml
```

- Register on Weights and Biases https://wandb.ai/site/, and get a key at https://wandb.ai/authorize. At some point, the script above (after processing the data) will ask if you have an account, enter yes, and then copy your key. Once the script finishes, look at the results on weights and biases website.

- Modify the network parameters in "runs/config_RP.yaml". Of importance are the number of epochs, the tau_pos_s value (10 second segment), the window_len_s (sub window of the segment, for example 2 seconds) and "n_samples_per_dataset" the number of sample to extract from each dataset (for example 100). This means that a window of 2 second will be extracted 100 times (at different latencies) from the 10 second segment.

