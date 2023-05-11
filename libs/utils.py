import torch
import mne
import numpy as np
import scipy
from matplotlib import pyplot as plt

class NNUtils():
    data_keys = ["feat_inst_theta", "feat_inst_alpha", "feat_inst_beta"] # R G B

    def __topomap_tensor_mne(self, data_rgb, mask):
        """
        Returns a 300x300x3 RGB topo data array projected onto a head model
        """
        figs = []

        for data, color in zip(data_rgb, ["Reds", "Greens", "Blues"]):
            mne.viz.plot_topomap(
                data, self.info, ch_type='eeg', sensors=False, contours=0,
                cmap=color, outlines='head', size=4, show=False)
            fig = plt.gcf()
            fig.canvas.draw()
            figs.append(fig)
            plt.close()

        # Convert to RGB array
        fig_data = np.array([np.asarray(fig.canvas.buffer_rgba())[:,:,n] for n,fig in enumerate(figs)])
        buffer = np.transpose(fig_data, axes=(1,2,0))

        if mask:
            mne.viz.plot_topomap(
                np.ones(32), self.info, ch_type='eeg', sensors=False, contours=0,
                cmap="Reds", outlines='head', size=4, show=False)
            fig_mask = plt.gcf()
            fig_mask.canvas.draw()
            plt.close()
            fig_mask_data = np.array(np.asarray(fig_mask.canvas.buffer_rgba())[:,:,0])
            mask = np.array((fig_mask_data == 103), dtype=np.uint8)
            buffer = buffer * np.expand_dims(mask, axis=-1)

        return buffer[50:-50,50:-50,:] # trim padding and return

    def generate_network_feat(self, data):
        # Assumes all data_keys have same dimensionality
        shape_time = data[self.data_keys[0]].shape[0]

        effective_window = shape_time if self.win is None else self.win
        effective_stride = effective_window if self.stride is None else self.stride

        start_idxs = np.arange(0, shape_time, effective_stride)
        stop_idxs = np.arange(effective_window, shape_time+1, effective_stride)
        range_tuples = zip(start_idxs[:len(stop_idxs)], stop_idxs)
        range_idxs = [np.arange(a,b) for a,b in range_tuples]

        d = [np.mean(
                np.take(data[k], range_idxs, axis=0), # SLICE WINDOW F
                axis=1,
                keepdims=False) # SLICE F
            for k in self.data_keys] # KEYS SLICE F

        d = np.transpose(d, (1,0,2)) # SLICE KEYS F
        tensor = [] # SLICE F..
        for i in range(d.shape[0]):
            tensor.append(self.__topomap_tensor_mne(d[i], mask=True)) # F..

        return np.array(tensor)

    def randomize_weights(self, rand_params):
        def zero_weights(m):
            # TODO this might need expansion depending on model
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                m.weight.data.fill_(0.)
                if hasattr(m, "bias"):
                    m.bias.data.fill_(0.)
        def init_weights(m):
            # TODO this might need expansion depending on model
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                m.reset_parameters()

        if rand_params["mode"] == "zero":
            self.model.apply(weight_reset)

        elif rand_params["mode"] == "rand_init":
            if "seed" in rand_params:
                torch.manual_seed(rand_params["seed"])
            self.model.apply(init_weights)

        elif rand_params["mode"] == "perturb":
            for idx, (name, param) in enumerate(self.model.named_parameters()):
                noise = torch.zeros_like(param)
                val_range = (torch.max(param)-torch.min(param)).item()
                if rand_params["distribution"] == "gaussian":
                    torch.nn.init.normal_(noise, 0, val_range/100)
                elif rand_params["distribution"] == "uniform":
                    torch.nn.init.uniform_(noise, val_range/200, val_range/100)
                else:
                    raise(f"Distribution {rand_params['distribution']} not supported")
                with torch.no_grad():
                    param += noise

        elif rand_params["mode"] == "shuffle":
            for idx, (name, param) in enumerate(self.model.named_parameters()):
                ori_shape = param.shape
                with torch.no_grad():
                    flattened = param.flatten()
                    flattened = flattened[torch.randperm(len(flattened))]
                    param.copy_(torch.reshape(flattened, ori_shape))

        else:
            raise ValueError(f"Unsupported random mode {rand_params['mode']}")
