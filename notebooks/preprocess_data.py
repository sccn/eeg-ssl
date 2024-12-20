import braindecode
from braindecode.datautil import load_concat_dataset
from braindecode.preprocessing import (
    preprocess, Preprocessor, create_fixed_length_windows)
from numpy import multiply

loaded_dataset = load_concat_dataset(path='data/hbn', preload=False)
all_ds = loaded_dataset
sampling_rate = 250 # resample to follow the tutorial sampling rate
high_cut_hz = 50
n_jobs = -1
# Factor to convert from V to uV
factor = 1e4
preprocessors = [
    Preprocessor(lambda data: multiply(data, factor)),  # Convert from V to uV
    Preprocessor('crop', tmin=10),  # crop first 10 seconds as begining of noise recording after visual inspection
    Preprocessor('resample', sfreq=sampling_rate),
    Preprocessor('notch_filter', freqs=(60, 120)),
    Preprocessor('filter', l_freq=None, h_freq=high_cut_hz, n_jobs=n_jobs)
]

# Transform the data
preprocess(all_ds, preprocessors)
all_ds.save('data/hbn_preprocessed', overwrite=True)