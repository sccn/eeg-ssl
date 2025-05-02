import torch
import torchvision.models as torchmodels
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
import random
import numpy as np
from math import ceil

class VGGSSL(nn.Module):
    def __init__(self):
        super().__init__()
        inchans = 129
        out_emb = 1024
        vgg = self.create_vgg_rescaled()
        self.model = nn.Sequential(
            vgg.features, 
            nn.Conv2d(64, 1, 1),
            nn.AdaptiveAvgPool2d(32),
            nn.Flatten(),
        )

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        z = self.model(x)
        return z
        
    def create_vgg_rescaled(self, subsample=4, feature='raw', weights='DEFAULT'):
        tmp = torchmodels.vgg16(weights=weights)
        tmp.features = tmp.features[0:17]
        vgg16_rescaled = nn.Sequential()
        modules = []
        
        if feature == 'raw':
            first_in_channels = 1
            first_in_features = 6144
        else:
            first_in_channels = 3
            first_in_features = 576
            
        for layer in tmp.features.children():
            if isinstance(layer, nn.Conv2d):
                if layer.in_channels == 3:
                    in_channels = first_in_channels
                else:
                    in_channels = int(layer.in_channels/subsample)
                out_channels = int(layer.out_channels/subsample)
                modules.append(nn.Conv2d(in_channels, out_channels, layer.kernel_size, layer.stride, layer.padding))
            else:
                modules.append(layer)
        vgg16_rescaled.add_module('features',nn.Sequential(*modules))
        vgg16_rescaled.add_module('flatten', nn.Flatten())

        modules = []
        for layer in tmp.classifier.children():
            if isinstance(layer, nn.Linear):
                if layer.in_features == 25088:
                    in_features = first_in_features
                else:
                    in_features = int(layer.in_features/subsample) 
                if layer.out_features == 1000:
                    out_features = 2
                else:
                    out_features = int(layer.out_features/subsample) 
                modules.append(nn.Linear(in_features, out_features))
            else:
                modules.append(layer)
        vgg16_rescaled.add_module('classifier', nn.Sequential(*modules))
        return vgg16_rescaled

class SpatialAttention(nn.Module):
    def __init__(self, out_channels=129, K=32):
        super().__init__()
        self.outchans = out_channels
        self.K = K       
        # trainable parameter:
        self.z = Parameter(torch.randn(self.outchans, K*K, dtype = torch.cfloat)/(K*K)) # each output channel has its own KxK z matrix
        self.z.requires_grad = True

    def forward(self, X, positions):
        '''
        @param X: BxCxT tensor
        @param positions: BxCx2 tensor
        '''
        def compute_cos_sin(x, y, K):
            '''
            Convert 2D x-y coordinates to frequency space backed by K frequencies
            '''
            kk = torch.arange(1, K+1)
            ll = torch.arange(1, K+1)
            cos_fun = lambda k, l, x, y: torch.cos(2*torch.pi*(k*x + l*y))
            sin_fun = lambda k, l, x, y: torch.sin(2*torch.pi*(k*x + l*y))
            return torch.stack([cos_fun(kk[None,:], ll[:,None], x, y) for x, y in zip(x, y)]).reshape(x.shape[0],-1).float(), \
                   torch.stack([sin_fun(kk[None,:], ll[:,None], x, y) for x, y in zip(x, y)]).reshape(x.shape[0],-1).float()
            
        sol = []   
        for i in len(X):
            eeg = X[i]
            x, y = positions[i, :, 0], positions[i, :, 1]
            cos_mat, sin_mat = compute_cos_sin(x, y, self.K)
            a = torch.matmul(self.z.real, cos_mat.T) + torch.matmul(self.z.imag, sin_mat.T)
            index = random.randint(0, len(x)-1)
            x_drop = x[index]
            y_drop = y[index]          
            # Question: divide this with square root of KxK? to stablize gradient as with self-attention?
            for i in range(a.shape[1]):
                distance = (x_drop - x)**2 + (y_drop - y)**2
                if distance < 0.1:
                    a = torch.cat((a[:, :i], a[:, i+1:]), dim = 1)
                    eeg = torch.cat((eeg[:i], eeg[i+1:]), dim=0)
            
            a = F.softmax(a, dim=1) # softmax over all input chan location for each output chan
                                                # outchans x  inchans
                    
            # X: N x 273 x 360            
            sol.append(torch.matmul(a, eeg)) # N x outchans x 360 (time)
                                    # matmul dim expansion logic: https://pytorch.org/docs/stable/generated/torch.matmul.html
        return torch.stack(sol)

class PosEmb(nn.Module):
  def __init__(self):
    super(PosEmb, self).__init__()
    self.conv = nn.Conv1d(768, 768, kernel_size=(128,), stride=(1,), padding=(64,), groups=16)
    self.conv = nn.utils.parametrizations.weight_norm(self.conv, name="weight", dim=2)
    self.activation = nn.GELU()
  def forward(self, x):
    x = self.conv(x)
    x = x[:, :, :-1]
    return torch.permute(self.activation(x), (0,2,1))

class Wav2Vec2(nn.Module):
  def __init__(self, K, S):
    super(Wav2Vec2, self).__init__()
    self.conv0 = []
    self.K = K
    self.S = S
    self.conv0.append(nn.Sequential(
        nn.Conv1d(128, 512, kernel_size=(self.K[0],), stride=(self.S[0],), bias=False),
        nn.GELU(),
        nn.GroupNorm(512, 512, eps=1e-05, affine=True)
    ))
    self.conv1 = []
    for i in range(4):
      self.conv1.append(nn.Sequential(
          nn.Conv1d(512, 512, kernel_size=(self.K[1],), stride=(self.S[1],), bias=False),
          nn.GELU()
      ))
    self.conv2 = []
    for i in range(2):
      self.conv2.append(nn.Sequential(
          nn.Conv1d(512, 512, kernel_size=(self.K[5],), stride=(self.S[5],), bias=False),
          nn.GELU()
        ))
    self.extractor = nn.Sequential(*self.conv0, *self.conv1, *self.conv2)
    self.projection = nn.Sequential(
        nn.LayerNorm((512,), eps=1e-05, elementwise_affine=True),
        nn.Linear(in_features=512, out_features=768, bias=True),
        nn.Dropout(p=0.1, inplace=False)
    )
    self.pos_emb = PosEmb()
    self.norm = nn.Sequential(
        nn.LayerNorm((768,), eps=1e-05, elementwise_affine=True),
        nn.Dropout(p=0.1, inplace=False)
    )
    self.attention = []
    for i in range(12):
      self.attention.append(nn.MultiheadAttention(768, 1))
    self.feed_forward = nn.Sequential(
        nn.Dropout(p=0.1, inplace=False),
        nn.Linear(in_features=768, out_features=3072, bias=True),
        nn.GELU(),
        nn.Linear(in_features=3072, out_features=768, bias=True),
        nn.Dropout(p=0.1, inplace=False),
        nn.LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        )


  def receptive_field(self):
    St = 1
    stride = self.S[::-1]
    for s in stride:
      St *= s
    R = 1
    i = 0
    kernel = self.K[::-1]
    for k in kernel:
      R = R*stride[i] + (k - stride[i])
      i += 1
    return R, St

  def feature_encoder(self, x):
    x = self.extractor(x)
    x = torch.permute(x, (0,2,1))
    return torch.permute(self.projection(x), (0,2,1))

  def context_encoder(self, x):
    x = self.pos_emb(x) + torch.permute(x, (0,2,1))
    x = self.norm(x)
    for i in range(12):
      x = self.attention[i](x, x, x)[0]
    return torch.permute(self.feed_forward(x), (0,2,1))

  def forward(self, x):
    x = self.feature_encoder(x)
    x = torch.flatten(x, start_dim = 2)
    x = self.context_encoder(x)
    return x

class Wav2VecBrainModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.ninput_channel = 128
        self.encoder_embed_dim = 768
        self.feature_encoder = self.FeatureEncoder(input_chan=self.ninput_channel)
        self.context_encoder = self.TransformerLayer(input_chan=self.ninput_channel)
        self.mask_emb = nn.Parameter(torch.FloatTensor(self.encoder_embed_dim).uniform_())

    def forward(self, x):
        x = self.feature_encoder(x)
        x = self.context_encoder(x)

        return x


    class FeatureEncoder(nn.Module):
        def __init__(self, input_chan):
            super().__init__()
            self.input_chan = input_chan
            self.K = [10, 3, 3, 3, 3, 2, 2]
            self.S = [2, 1, 1, 1, 1, 1, 1]
            self.conv0 = []
            self.conv0.append(nn.Sequential(
                nn.Conv1d(self.input_chan, 512, kernel_size=(10,), stride=(4,), bias=False),
                nn.GELU(),
                nn.GroupNorm(512, 512, eps=1e-05, affine=True)
            ))
            self.conv1 = []
            for i in range(4):
                self.conv1.append(nn.Sequential(
                    nn.Conv1d(512, 512, kernel_size=(3,), stride=(1,), bias=False),
                    nn.GELU()
                ))
            self.conv2 = []
            for i in range(2):
                self.conv2.append(nn.Sequential(
                    nn.Conv1d(512, 512, kernel_size=(2,), stride=(1,), bias=False),
                    nn.GELU()
                    ))
            self.extractor = nn.Sequential(*self.conv0, *self.conv1, *self.conv2)
            self.projection = nn.Sequential(
                nn.LayerNorm((512,), eps=1e-05, elementwise_affine=True),
                nn.Linear(in_features=512, out_features=768, bias=True),
                nn.Dropout(p=0.1, inplace=False)
            )

        def receptive_field(self):
            St = 1
            stride = self.S[::-1]
            for s in stride:
                St *= s
            R = 1
            i = 0
            kernel = self.K[::-1]
            for k in kernel:
                R = R*stride[i] + (k - stride[i])
                i += 1
            return R, St

        def forward(self, x):
            x = self.extractor(x)
            x = torch.permute(x, (0,2,1))
            return torch.permute(self.projection(x), (0,2,1))

    class TransformerLayer(nn.Module):
        class PosEmb(nn.Module):
            def __init__(self, input_chan):
                super().__init__()
                self.conv = nn.Conv1d(768, 768, kernel_size=(input_chan,), stride=(1,), padding=(64,), groups=16)
                self.conv = nn.utils.parametrizations.weight_norm(self.conv, name="weight", dim=2)
                self.activation = nn.GELU()
            def forward(self, x):
                x = self.conv(x)
                x = x[:, :, :-1]
                return torch.permute(self.activation(x), (0,2,1))

        def __init__(self, input_chan):
            super().__init__()
            self.pos_emb = self.PosEmb(input_chan)
            self.norm = nn.Sequential(
                nn.LayerNorm((768,), eps=1e-05, elementwise_affine=True),
                nn.Dropout(p=0.1, inplace=False)
            )
            self.attention = nn.Sequential(*[nn.MultiheadAttention(768, 1) for i in range(12)])
            self.feed_forward = nn.Sequential(
                nn.Dropout(p=0.1, inplace=False),
                nn.Linear(in_features=768, out_features=3072, bias=True),
                nn.GELU(),
                nn.Linear(in_features=3072, out_features=768, bias=True),
                nn.Dropout(p=0.1, inplace=False),
                nn.LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                )
        
        def forward(self, x):
            x = self.pos_emb(x) + torch.permute(x, (0,2,1))
            x = self.norm(x)
            for i in range(12):
                x = self.attention[i](x, x, x)[0]
            return torch.permute(self.feed_forward(x), (0,2,1))

class LacunaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_encoder = self.build_feature_encoder()
        self.context_encoder = nn.LSTM(630, 512, 3, bidirectional= True, batch_first=True)

    def build_feature_encoder(self):
        conv0 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size = (4, 2), stride = (4, 2)),
            nn.LeakyReLU(),
            nn.GroupNorm(64,64)
        )
        conv1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size = (4, 2)),
            nn.LeakyReLU(),
            nn.GroupNorm(128,128)
        )
        conv2 = nn.Sequential(
            nn.MaxPool2d((4, 2), stride = (1, 2)),
            nn.Conv2d(128, 256, kernel_size = (3, 3)),
            nn.MaxPool2d((4, 2), stride = (1, 2))
        )
        return nn.Sequential(
            conv0,
            conv1,
            conv2
        )

    def forward(self, x):
        x = self.feature_encoder(x)
        x = torch.flatten(x, start_dim = 2)
        x = self.context_encoder(x)
        return x

############### BENDR model ######################
import copy
import torch
from torch import nn
from einops.layers.torch import Rearrange
from braindecode.models.base import EEGModuleMixin
class BENDR(EEGModuleMixin, nn.Module):
    def __init__(
        self,
        # Signal related parameters (required by EEGModuleMixin or for setup)
        n_chans=None,  # Number of EEG channels
        n_outputs=None,  # Number of outputs for the final layer
        n_times=None,  # Number of time points in input
        chs_info=None,  # Info about channels (optional from braindecode)
        input_window_seconds=None,  # Duration of input (optional from braindecode)
        sfreq=None,  # Sampling frequency (optional from braindecode)
        # Model parameters
        encoder_h=512,  # Hidden size of the encoder convolutional layers
        contextualizer_hidden=3076,  # Feedforward hidden size in transformer
        projection_head=False,  # Whether encoder should project back to input feature size (unused in original fine-tuning)
        drop_prob=0.1,  # General dropout probability
        layer_drop=0.0,  # Probability of dropping transformer layers during training
        activation=nn.GELU,  # Activation function
        # Transformer specific parameters (add defaults matching original if needed)
        transformer_layers=8,
        transformer_heads=8,
        position_encoder_length=25,  # Kernel size for positional encoding conv
        enc_width=(3, 2, 2, 2, 2, 2),
        enc_downsample=(3, 2, 2, 2, 2, 2),
        # extra model parameters
        start_token=-5,  # Value for start token embedding
        final_layer=True,  # Whether to include the final linear layer
    ):
        # Initialize EEGModuleMixin first if it provides n_chans, n_outputs etc.
        # Ensure required parameters like n_chans, n_outputs are set before use
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )

        # Keep these parameters if needed later, otherwise they are captured by the mixin
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq

        self.encoder_h = encoder_h
        self.contextualizer_hidden = contextualizer_hidden
        self.include_final_layer = final_layer  # Renamed to avoid conflict

        # Encoder: Use parameters from __init__
        self.encoder = ConvEncoderBENDR(
            in_features=self.n_chans,  # Use n_chans from mixin/init
            encoder_h=encoder_h,
            dropout=drop_prob,
            projection_head=projection_head,  # Pass projection_head parameter
            enc_width=enc_width,
            enc_downsample=enc_downsample,
        )

        # Contextualizer: Use parameters from __init__
        self.contextualizer = BENDRContextualizer(
            in_features=self.encoder.encoder_h,  # Use the output feature size of the encoder
            hidden_feedforward=contextualizer_hidden,
            heads=transformer_heads,
            layers=transformer_layers,
            dropout=drop_prob,  # Use general dropout probability
            # activation="gelu", # Pass activation name string
            position_encoder=position_encoder_length,  # Pass position encoder kernel size
            layer_drop=layer_drop,
            start_token=start_token,  # Keep fixed start token value
        )
        in_features = self.encoder.encoder_h  # Set in_features for final layer
        # Final Layer: Use LazyLinear to adapt to input size automatically
        self.final_layer = None  # Initialize
        if self.include_final_layer:
            # Input to LazyLinear will be [batch_size, encoder_h] after taking last timestep
            linear = nn.Linear(in_features=in_features, out_features=self.n_outputs)
            self.final_layer = nn.utils.parametrizations.weight_norm(
                linear, name="weight", dim=1
            )

    def forward(self, x):
        # Input x: [batch_size, n_chans, n_times]
        encoded = self.encoder(x)
        # encoded: [batch_size, encoder_h, n_encoded_times]

        context = self.contextualizer(encoded)
        # context: [batch_size, encoder_h, n_encoded_times + 1] (due to start token)

        # Extract features - take the state corresponding to the *last input timestep*
        # The output has shape [batch_size, features, seq_len+1]
        # The last element context[:,:,-1] corresponds to the *last input time step*
        # (assuming start token is at index 0 after permute in contextualizer)
        # However, often the output corresponding to the *start token* (index 0) is used
        # as the aggregate representation. Let's assume you want the last input timestep's state.
        # Check the transformer's output dimensions carefully based on start_token handling.
        # If output is [batch_size, features, seq_len+1], then last *input* timestep is -1
        feature = context[:, :, -1]
        # feature: [batch_size, encoder_h]

        if self.final_layer is not None:
            feature = self.final_layer(feature)
            # feature: [batch_size, n_outputs]

        return feature

class _BENDREncoder(nn.Module):
    def __init__(self, in_features, encoder_h=256,):
        super().__init__()
        self.in_features = in_features
        self.encoder_h = encoder_h

    def load(self, filename, strict=True):
        state_dict = torch.load(filename)
        self.load_state_dict(state_dict, strict=strict)

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def freeze_features(self, unfreeze=False):
        for param in self.parameters():
            param.requires_grad = unfreeze

class ConvEncoderBENDR(_BENDREncoder):
    def __init__(self, in_features, encoder_h=256, enc_width=(3, 2, 2, 2, 2, 2),
                 dropout=0., projection_head=False, enc_downsample=(3, 2, 2, 2, 2, 2)):
        super().__init__(in_features, encoder_h)
        self.encoder_h = encoder_h
        if not isinstance(enc_width, (list, tuple)):
            enc_width = [enc_width]
        if not isinstance(enc_downsample, (list, tuple)):
            enc_downsample = [enc_downsample]
        assert len(enc_downsample) == len(enc_width)

        # Centerable convolutions make life simpler
        enc_width = [e if e % 2 else e+1 for e in enc_width]
        self._downsampling = enc_downsample
        self._width = enc_width

        self.encoder = nn.Sequential()
        for i, (width, downsample) in enumerate(zip(enc_width, enc_downsample)):
            self.encoder.add_module("Encoder_{}".format(i), nn.Sequential(
                nn.Conv1d(in_features, encoder_h, width, stride=downsample, padding=width // 2),
                nn.Dropout1d(dropout), # warning
                nn.GroupNorm(encoder_h // 2, encoder_h),
                nn.GELU(),
            ))
            in_features = encoder_h

        if projection_head:
            self.encoder.add_module("projection-1", nn.Sequential(
                nn.Conv1d(in_features, in_features, 1),
                nn.Dropout2d(dropout*2),
                nn.GroupNorm(in_features // 2, in_features),
                nn.GELU()
            ))

    def description(self, sfreq=None, sequence_len=None):
        widths = list(reversed(self._width))[1:]
        strides = list(reversed(self._downsampling))[1:]

        rf = self._width[-1]
        for w, s in zip(widths, strides):
            rf = rf if w == 1 else (rf - 1) * s + 2 * (w // 2)

        desc = "Receptive field: {} samples".format(rf)
        if sfreq is not None:
            desc += ", {:.2f} seconds".format(rf / sfreq)

        ds_factor = np.prod(self._downsampling)
        desc += " | Downsampled by {}".format(ds_factor)
        if sfreq is not None:
            desc += ", new sfreq: {:.2f} Hz".format(sfreq / ds_factor)
        desc += " | Overlap of {} samples".format(rf - ds_factor)
        if sequence_len is not None:
            desc += " | {} encoded samples/trial".format(sequence_len // ds_factor)
        return desc

    def downsampling_factor(self, samples):
        for factor in self._downsampling:
            samples = ceil(samples / factor)
        return samples

    def forward(self, x):
        return self.encoder(x)

class _Hax(nn.Module):
    """T-fixup assumes self-attention norms are removed"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class Permute(nn.Module):
    def __init__(self, axes):
        super().__init__()
        self.axes = axes

    def forward(self, x):
        return x.permute(self.axes)

def _make_span_from_seeds(seeds, span, total=None):
    inds = list()
    for seed in seeds:
        for i in range(seed, seed + span):
            if total is not None and i >= total:
                break
            elif i not in inds:
                inds.append(int(i))
    return np.array(inds)

def _make_mask(shape, p, total, span, allow_no_inds=False):
    # num_mask_spans = np.sum(np.random.rand(total) < p)
    # num_mask_spans = int(p * total)
    mask = torch.zeros(shape, requires_grad=False, dtype=torch.bool)

    for i in range(shape[0]):
        mask_seeds = list()
        while not allow_no_inds and len(mask_seeds) == 0 and p > 0:
            mask_seeds = np.nonzero(np.random.rand(total) < p)[0]

        mask[i, _make_span_from_seeds(mask_seeds, span, total=total)] = True

    return mask

class BENDRContextualizer(nn.Module):

    def __init__(self, in_features, hidden_feedforward=3076, heads=8, layers=8, dropout=0.15, activation='gelu',
                 position_encoder=25, layer_drop=0.0, mask_p_t=0.1, mask_p_c=0.004, mask_t_span=6, mask_c_span=64,
                 start_token=-5, finetuning=False):
        super().__init__()

        self.dropout = dropout
        self.in_features = in_features
        self._transformer_dim = in_features * 3

        encoder = nn.TransformerEncoderLayer(d_model=in_features * 3, nhead=heads, dim_feedforward=hidden_feedforward,
                                             dropout=dropout, activation=activation)
        encoder.norm1 = _Hax()
        encoder.norm2 = _Hax()

        self.norm = nn.LayerNorm(self._transformer_dim)

        # self.norm_layers = nn.ModuleList([copy.deepcopy(norm) for _ in range(layers)])
        self.transformer_layers = nn.ModuleList([copy.deepcopy(encoder) for _ in range(layers)])
        self.layer_drop = layer_drop
        self.p_t = mask_p_t
        self.p_c = mask_p_c
        self.mask_t_span = mask_t_span
        self.mask_c_span = mask_c_span
        self.start_token = start_token
        self.finetuning = finetuning

        # Initialize replacement vector with 0's
        self.mask_replacement = torch.nn.Parameter(torch.normal(0, in_features**(-0.5), size=(in_features,)),
                                                   requires_grad=True)

        self.position_encoder = position_encoder > 0
        if position_encoder:
            conv = nn.Conv1d(in_features, in_features, position_encoder, padding=position_encoder // 2, groups=16)
            nn.init.normal_(conv.weight, mean=0, std=2 / self._transformer_dim)
            nn.init.constant_(conv.bias, 0)
            conv = nn.utils.parametrizations.weight_norm(conv, dim=2)
            self.relative_position = nn.Sequential(conv, nn.GELU())

        self.input_conditioning = nn.Sequential(
            Permute([0, 2, 1]),
            nn.LayerNorm(in_features),
            nn.Dropout(dropout),
            Permute([0, 2, 1]),
            nn.Conv1d(in_features, self._transformer_dim, 1),
            Permute([2, 0, 1]),
        )

        self.output_layer = nn.Conv1d(self._transformer_dim, in_features, 1)
        self.apply(self.init_bert_params)

    def init_bert_params(self, module):
        if isinstance(module, nn.Linear):
            # module.weight.data.normal_(mean=0.0, std=0.02)
            nn.init.xavier_uniform_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()
            # Tfixup
            module.weight.data = 0.67 * len(self.transformer_layers) ** (-0.25) * module.weight.data

        # if isinstance(module, nn.Conv1d):
        #     # std = np.sqrt((4 * (1.0 - self.dropout)) / (self.in_features * self.in_features))
        #     # module.weight.data.normal_(mean=0.0, std=std)
        #     nn.init.xavier_uniform_(module.weight.data)
        #     module.bias.data.zero_()

    def forward(self, x, mask_t=None, mask_c=None):
        bs, feat, seq = x.shape
        if self.training and self.finetuning:
            if mask_t is None and self.p_t > 0:
                mask_t = _make_mask((bs, seq), self.p_t, x.shape[-1], self.mask_t_span)
            if mask_c is None and self.p_c > 0:
                mask_c = _make_mask((bs, feat), self.p_c, x.shape[1], self.mask_c_span)

        if mask_t is not None:
            x.transpose(2, 1)[mask_t] = self.mask_replacement
        if mask_c is not None:
            x[mask_c] = 0

        if self.position_encoder:
            x = x + self.relative_position(x)
        x = self.input_conditioning(x)

        if self.start_token is not None:
            in_token = self.start_token * torch.ones((1, 1, 1), requires_grad=True).to(x.device).expand([-1, *x.shape[1:]])
            x = torch.cat([in_token, x], dim=0)

        for layer in self.transformer_layers:
            if not self.training or torch.rand(1) > self.layer_drop:
                x = layer(x)

        return self.output_layer(x.permute([1, 2, 0]))

    def freeze_features(self, unfreeze=False, finetuning=False):
        for param in self.parameters():
            param.requires_grad = unfreeze
        if self.finetuning or finetuning:
            self.mask_replacement.requires_grad = False

    def load(self, filename, strict=True):
        state_dict = torch.load(filename)
        self.load_state_dict(state_dict, strict=strict)

    def save(self, filename):
        torch.save(self.state_dict(), filename)