import torch
import torchvision.models as torchmodels
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
import random

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
    self.conv = nn.utils.weight_norm(self.conv, name="weight", dim=2)
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
                self.conv = nn.utils.weight_norm(self.conv, name="weight", dim=2)
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
        self.encoder = _ConvEncoderBENDR(
            in_features=self.n_chans,  # Use n_chans from mixin/init
            encoder_h=encoder_h,
            dropout=drop_prob,
            projection_head=projection_head,  # Pass projection_head parameter
            enc_width=enc_width,
            enc_downsample=enc_downsample,
        )

        # Contextualizer: Use parameters from __init__
        self.contextualizer = _BENDRContextualizer(
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


class _ConvEncoderBENDR(nn.Module):
    def __init__(
        self,
        in_features,
        encoder_h=512,
        enc_width=(3, 2, 2, 2, 2, 2),
        dropout=0.0,
        projection_head=False,
        enc_downsample=(3, 2, 2, 2, 2, 2),
    ):
        super().__init__()
        self.encoder_h = encoder_h
        self.in_features = in_features

        if not isinstance(enc_width, (list, tuple)):
            enc_width = [enc_width]
        if not isinstance(enc_downsample, (list, tuple)):
            enc_downsample = [enc_downsample]
        if len(enc_downsample) != len(enc_width):
            raise ValueError(
                "Encoder width and downsampling factors must have the same length."
            )

        # Centerable convolutions make life simpler
        enc_width = [
            e if e % 2 != 0 else e + 1 for e in enc_width
        ]  # Ensure odd kernel size
        self._downsampling = enc_downsample
        self._width = enc_width

        current_in_features = in_features
        self.encoder = nn.Sequential()
        for i, (width, downsample) in enumerate(zip(enc_width, enc_downsample)):
            self.encoder.add_module(
                "Encoder_{}".format(i),
                nn.Sequential(
                    nn.Conv1d(
                        current_in_features,
                        encoder_h,
                        width,
                        stride=downsample,
                        padding=width
                        // 2,  # Correct padding for 'same' output length before stride
                    ),
                    nn.Dropout1d(dropout),  # 1D dropout for 1D conv
                    nn.GroupNorm(
                        encoder_h // 2, encoder_h
                    ),  # Consider making num_groups configurable or ensure encoder_h is divisible by 2
                    nn.GELU(),
                ),
            )
            current_in_features = encoder_h  # Update in_features for the next layer

        if projection_head:
            self.encoder.add_module(
                "projection_head",
                nn.Conv1d(
                    encoder_h, encoder_h, 1
                ),  # Project back to encoder_h or in_features? BENDR paper implies just conv layers.
                # Original uses more complex projection in LinearHeadBENDR's EncodingAugment
            )

    def forward(self, x):
        return self.encoder(x)


class _BENDRContextualizer(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_feedforward=3076,
        heads=8,
        layers=8,
        dropout=0.1,  # Default dropout
        activation="gelu",  # Activation for transformer FF layer
        position_encoder=25,  # Kernel size for conv positional encoding
        layer_drop=0.0,  # Probability of dropping a whole layer
        start_token=-5,  # Value for start token embedding
        # finetuning=False, # This flag existed in original code, might control masking behaviour (removed here)
    ):
        super().__init__()
        self.dropout = dropout
        self.layer_drop = layer_drop
        self.start_token = start_token  # Store start token value

        # The input dimension to the transformer layers
        # Original BENDR uses 3 * in_features. Let's stick to in_features for simplicity first,
        # unless the 3* dimension is critical. The paper suggests a projection up occurs.
        # Let's follow the original's projection:
        self.transformer_dim = in_features
        self.in_features = in_features
        # --- Positional Encoding --- (Applied before projection)
        self.position_encoder = None
        if position_encoder > 0:
            conv = nn.Conv1d(
                in_features,
                in_features,
                kernel_size=position_encoder,
                padding=position_encoder // 2,
                groups=16,  # Number of groups for depthwise separation
            )
            # Initialize weights first
            nn.init.normal_(conv.weight, mean=0, std=0.02)  # Basic init
            nn.init.constant_(conv.bias, 0)

            conv = nn.utils.parametrizations.weight_norm(conv, name="weight", dim=2)
            self.relative_position = nn.Sequential(conv, nn.GELU())
        # --- Input Conditioning --- (Includes projection up to transformer_dim)
        # Rearrange, Norm, Dropout, Project, Rearrange
        self.input_conditioning = nn.Sequential(
            Rearrange("b c t -> b t c"),  # Batch, Time, Channels
            nn.LayerNorm(in_features),
            nn.Dropout(dropout),
            nn.Linear(in_features, self.transformer_dim),  # Project up using Linear
            # nn.Conv1d(in_features, self.transformer_dim, 1), # Alternative: Project up using Conv1d
            Rearrange(
                "b t c -> t b c"
            ),  # Time, Batch, Channels (Transformer expected format)
        )

        # --- Transformer Encoder Layers ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.transformer_dim,  # Use projected dimension
            nhead=heads,
            dim_feedforward=hidden_feedforward,
            dropout=dropout,  # Dropout within transformer layer
            activation=activation,
            batch_first=False,  # Expects (T, B, C)
            norm_first=False,  # Standard post-norm architecture
        )
        self.transformer_layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(layers)]
        )

        # --- Output Layer --- (Project back down to in_features)
        # Input is (T, B, C_transformer), output should be (B, C_original, T)
        self.output_layer = nn.Linear(self.transformer_dim, in_features)

        # Initialize parameters like BERT / TFixup
        self.apply(self._init_simplified_params)

    def _init_bert_params(self, module):
        """Initialize linear layers and apply TFixup scaling."""
        if isinstance(module, nn.Linear):
            # Standard init
            # module.weight.data.normal_(mean=0.0, std=0.02)
            nn.init.xavier_uniform_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()
            # Tfixup Scaling
            module.weight.data = (
                0.67 * len(self.transformer_layers) ** (-0.25) * module.weight.data
            )
        # You might want to initialize LayerNorm layers as well
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _init_simplified_params(self, module):
        """Initialize linear layers with Xavier and LayerNorms with defaults."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        # Input x: [batch_size, in_features, seq_len]

        # Apply relative positional encoding
        if hasattr(self, "relative_position"):
            pos_enc = self.relative_position(x)
            x = x + pos_enc

        # Apply input conditioning (includes projection up and rearrange)
        x = self.input_conditioning(x)
        # x: [seq_len, batch_size, transformer_dim]

        # Prepend start token
        if self.start_token is not None:
            token_emb = torch.full(
                (1, x.shape[1], x.shape[2]),
                float(self.start_token),
                device=x.device,
                requires_grad=False,
            )
            x = torch.cat([token_emb, x], dim=0)
        # x: [seq_len + 1, batch_size, transformer_dim]

        # Apply transformer layers with layer drop
        for layer in self.transformer_layers:
            if not self.training or torch.rand(1) > self.layer_drop:
                x = layer(x)
        # x: [seq_len + 1, batch_size, transformer_dim]

        # Apply final projection back to original feature dimension
        x = self.output_layer(x)
        # x: [seq_len + 1, batch_size, in_features]

        # Rearrange back to [batch_size, in_features, seq_len + 1]
        x = x.permute(1, 2, 0)

        return x