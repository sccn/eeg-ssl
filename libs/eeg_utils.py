import mne
from mne.time_frequency import psd_array_welch
from matplotlib import pyplot as plt
import numpy as np
import scipy.io
import os

# Allows access using . notation
# class EEG:
#     def __init__(self, **kwargs):
#         self.__dict__.update(kwargs)
#     def __getitem__(self, key):
#         return self.__dict__[key]
#     def __setitem__(self, key, value):
#         self.__dict__[key] = value

def plot_raw_eeg(data, sampling_freq, num_channels='all', channels=[]):
    '''
    Plot raw eeg data
    @params
        data - C x T
        channels: list of channel indices to plot
    '''
    C, T = data.shape
    print(f'Data shape: C, T: {C}, {T}')
    # create a stacked plot of EEG trace using matplotlib
    plt.figure()
    if channels:
        nchans =  len(channels)
    else:
        nchans = data.shape[0] if num_channels == 'all' else num_channels
        channels = range(nchans)

    for i, ch in enumerate(channels):
        plt.subplot(nchans, 1, i+1)
        plt.plot(np.arange(data.shape[1]), data[ch,:])
        plt.yticks([])
        plt.xticks([])
        plt.box(False)
    info = mne.create_info(nchans, sfreq=sampling_freq, ch_types='eeg')
    simulated_raw = mne.io.RawArray(data[channels,:], info)
    # simulated_raw.plot(show_scrollbars=False, show_scalebars=True, show=True)

    # Calculate PSD
    fmin, fmax = 0, sampling_freq/2-1  # Frequency range
    # Plot PSD
    simulated_raw.plot_psd(fmin=fmin, fmax=fmax, average=True, show=True)

def pop_loadset(file_path):
    # Load MATLAB file
    EEG = scipy.io.loadmat(file_path, struct_as_record=False, squeeze_me=True)
        
    def check_keys(dict_data):
        """
        Check if entries in dictionary are mat-objects. If yes,
        _to_dict is called to change them to dictionaries.
        Recursively go through the entire structure.
        """
        for key in dict_data:
            if isinstance(dict_data[key], scipy.io.matlab.mat_struct):
                dict_data[key] = to_dict(dict_data[key])
            elif isinstance(dict_data[key], dict):
                dict_data[key] = check_keys(dict_data[key])
            elif isinstance(dict_data[key], np.ndarray) and dict_data[key].dtype == object:
                dict_data[key] = np.array([check_keys({i: item})[i] if isinstance(item, dict) else item for i, item in enumerate(dict_data[key])], dtype=object)
                if dict_data[key].size == 0:
                    dict_data[key] = None
        return dict_data

    def to_dict(matobj):
        """
        A recursive function which constructs from matobjects nested dictionaries.
        """
        dict_data = {}
        for strg in matobj._fieldnames:
            elem = getattr(matobj, strg)
            if isinstance(elem, scipy.io.matlab.mat_struct):
                dict_data[strg] = to_dict(elem)
            elif isinstance(elem, np.ndarray) and elem.dtype == object:
                dict_data[strg] = np.array([to_dict(sub_elem) if isinstance(sub_elem, scipy.io.matlab.mat_struct) else sub_elem for sub_elem in elem], dtype=object)
                if dict_data[strg].size == 0:
                    dict_data[strg] = None
            else:
                dict_data[strg] = elem
        # check if contains empty arrays
        for key in dict_data:
            if isinstance(dict_data[key], np.ndarray) and dict_data[key].size == 0:
                dict_data[key] = np.array([])
                
        return dict_data

    # check if EEG['data'] is a string, and if it the case, read the binary float32 file
    EEG = check_keys(EEG)
    if 'EEG' in EEG:
        EEG = EEG['EEG']
        
    if 'data' in EEG and isinstance(EEG['data'], str):
        file_name = EEG['filepath'] + os.sep + EEG['data']
        EEG['data'] = np.fromfile(file_name, dtype='float32').reshape( EEG['pnts']*EEG['trials'], EEG['nbchan'])
        EEG['data'] = EEG['data'].T.reshape(EEG['nbchan'], EEG['trials'], EEG['pnts']).transpose(0, 2, 1)

    # compute ICA activations
    if 'icaweights' in EEG and 'icasphere' in EEG:
        EEG['icaact'] = np.dot(np.dot(EEG['icaweights'], EEG['icasphere']), EEG['data'].reshape(EEG['nbchan'], -1))
        EEG['icaact'] = EEG['icaact'].reshape(EEG['icaweights'].shape[0], -1, EEG['trials'])
            
    return EEG