Current objective is to apply the framework of King et al. 2024 to The Present movie on HBN releases on NEMAR.

# TODO
- [ ] Optimize data format: loading eeglab .set without .fdt in mne load the entire data. Might have to save to another format. Or update mne-python function to be able to load info only from single .set file. Ask Arno
- [ ] Integrate braindecode to this repo
    - [ ] Verify channel location extraction and 2D projection are correct
    - [ ] Optimize mne.Raw loading with lazy loading
    - [ ] Verify that all HBN NEMAR releases can be loaded
- [ ] Handle movie frames
    - [ ] Upload movie frame images to the repo
    - [ ] Explore image embedding models
    - [ ] Optionally format the annotations in a DataFrame to be used in the future
- [ ] Set up training pipeline following the paper
- [ ] Discuss training methods for our dataset