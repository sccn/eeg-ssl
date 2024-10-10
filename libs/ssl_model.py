import torch
import torchvision.models as torchmodels
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from types import SimpleNamespace
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, random_split
import os

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


class SSLModel(ABC, nn.Module):
    def __init__(self, model_params=None):
        super().__init__()
        default_params = {
            'task': 'RP',
            'weights': 'DEFAULT'
        }

        if model_params:
            default_params.update(model_params)
        for k,v in default_params.items():
            setattr(self, k, v)

    @abstractmethod
    def encode(self, x):
        pass

    @abstractmethod
    def aggregate(self, x):
        pass

class Wav2VecBrainModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.ninput_channel = 128
        self.encoder_embed_dim = 768
        self.feature_encoder = self.FeatureEncoder(input_chan=self.ninput_channel)
        self.context_encoder = self.TransformerLayer()
        self.mask_emb = nn.Parameter(torch.FloatTensor(self.encoder_embed_dim).uniform_())

    def forward(self, x):
        x = self.feature_encoder(x)
        x = self.context_encoder(x)

        return x


    class FeatureEncoder(nn.Module):
        def __init__(self, input_chan):
            super().__init__()
            self.input_chan = 128
            self.K = [10, 3, 3, 3, 3, 2, 2]
            self.S = [2, 1, 1, 1, 1, 1, 1]
            self.conv0 = []
            self.conv0.append(nn.Sequential(
                nn.Conv1d(self.ninput_channel, 512, kernel_size=(10,), stride=(4,), bias=False),
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
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv1d(768, 768, kernel_size=(self.ninput_channel,), stride=(1,), padding=(64,), groups=16)
                self.conv = nn.utils.weight_norm(self.conv, name="weight", dim=2)
                self.activation = nn.GELU()
            def forward(self, x):
                x = self.conv(x)
                x = x[:, :, :-1]
                return torch.permute(self.activation(x), (0,2,1))

        def __init__(self):
            super().__init__()
            self.pos_emb = self.PosEmb()
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

class VGGSSL(SSLModel):
    def __init__(self, model_params=None):
        super().__init__(model_params)
        self.encoder = self.create_vgg_rescaled(weights=self.weights)
        self.__model_augment()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
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

    def __model_augment(self):
        self.encoder = torch.nn.Sequential(self.encoder.features, self.encoder.flatten, nn.Linear(32768, 4096))
        if self.task == "CPC":
            self.gAR = nn.GRU(4096, 100) # hidden size = 100, per (Banville et al, 2020) experiment

    def forward(self, x):
        '''
        @param x: (batch_size, channel, time)
        '''
        x = x.unsqueeze(1)
        return self.encode(x)
    
    def encode(self, x):
        return self.encoder(x)

    def aggregate(self, x):
        return super().aggregate(x)
