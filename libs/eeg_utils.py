import mne
import numpy as np

def plot_raw_eeg(data):
    sampling_freq = 128  # in Hertz
    info = mne.create_info(data.shape[0], sfreq=sampling_freq)
    simulated_raw = mne.io.RawArray(data, info)
    simulated_raw.plot(show_scrollbars=False, show_scalebars=False)
