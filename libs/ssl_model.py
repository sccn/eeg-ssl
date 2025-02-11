import torch
import torchvision.models as torchmodels
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from torch import nn, optim
from torch.nn.parameter import Parameter
import lightning as L
from .evaluation import train_regressor, RankMe
import random

class LitSSL(L.LightningModule):
    def __init__(self, 
        encoder: nn.Module,
        encoder_emb_size=1024,
        emb_size=100, 
        dropout=0.5
    ):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = encoder
        encoder_expected_emb_size = 1024
        if encoder_emb_size != encoder_expected_emb_size:
            projection_layer = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(encoder_emb_size, encoder_expected_emb_size),
            )
        else:
            projection_layer = nn.Identity()
            
        self.embedder = nn.Sequential(
            self.encoder,
            projection_layer,
            nn.Dropout(dropout),
            nn.Linear(encoder_expected_emb_size, emb_size),
            nn.Dropout(dropout)
        )
            
        self.clf = nn.Linear(emb_size, 1)
        
        self.rankme = RankMe()
        
    def embed(self, x):
        return self.embedder(x)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        X, y = batch
        x1, x2 = X[0], X[1]
        z1, z2 = self.embed(x1), self.embed(x2)
        z = torch.abs(z1 - z2)
        loss = nn.functional.binary_cross_entropy_with_logits(self.clf(z).flatten(), y)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        X, Y, _ = batch
        z = self.clf(self.embed(X))
        self.rankme.update(z)
        
    def test_step(self, batch, batch_idx):
        # this is the test loop
        X, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        test_loss = F.mse_loss(x_hat, x)
        self.log("test_loss", test_loss)

    def on_validation_epoch_end(self):
        # log epoch metric
        self.log('val_rankme', self.rankme.compute(), sync_dist=True)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

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

