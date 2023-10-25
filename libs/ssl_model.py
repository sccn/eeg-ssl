import torch
import torchvision.models as torchmodels
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from types import SimpleNamespace
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, random_split
import os

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
    def classify(self, x):
        pass


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
        self.encoder = torch.nn.Sequential(self.encoder.features, self.encoder.flatten, nn.Linear(16384, 4096))
        if self.task == "RP":
            self.classifier = nn.Linear(4096, 2)
        elif self.task == "TS":
            self.classifier = nn.Linear(4096*2, 2)
        elif self.task == "CPC":
            self.gAR = nn.GRU(4096, 100) # hidden size = 100, per (Banville et al, 2020) experiment

    def forward(self, x):
        return self(x)
    
    def encode(self, x):
        return self.encoder(x)

    def classify(self, x):
        return self.classifier(x)
    
class SSLModelUtils():
    def __init__(self,
            model_params={
                'model': 'VGGSSL',
                'task': 'RP'
            },
            train_params={
                'batch_size': 16,
                'learning_rate': 0.001,
                'optimizer': None,                    
                'num_epochs': 500,
                'start_from': 0,
                'checkpoint_path': './checkpoints',
                'log_dir': './runs',
                'early_stopping_eps': 0.01,
                'lr_decay_nepoch': 100,
                'print_every': 10,
            }):
        self.task = model_params['task']
        self.model = globals()[model_params['model']](model_params)
        self.__init_train(train_params)
    
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __init_train(self, custom_params):
        '''
        Initialize training parameters
        '''
        train_params = {
            'batch_size': 16,
            'learning_rate': 0.001,
            'optimizer': None,                    
            'num_epochs': 500,
            'start_from': 0,
            'checkpoint_path': './checkpoints',
            'log_dir': './runs',
            'early_stopping_eps': 0.01,
            'lr_decay_nepoch': 100,
            'print_every': 10,
        }
        train_params.update(custom_params)
        self.train_params = train_params

        if not os.path.exists(self.train_params['checkpoint_path']):
            os.mkdir(self.train_params['checkpoint_path'])
        if not os.path.exists(self.train_params['log_dir']):
            os.mkdir(self.train_params['log_dir'])
        if not self.train_params['optimizer']:
            self.train_params['optimizer'] = torch.optim.Adamax(self.model.parameters(), lr = self.train_params['learning_rate'])
    
    def get_dataloader(self, data, shuffle=True):
        if type(data) == list:
            X = [i[0] for i in data]
            Y = [i[1] for i in data]
            loader = DataLoader(BaseDataset(X, Y), batch_size = self.train_params['batch_size'], shuffle = shuffle)
        elif isinstance(data, Dataset):
            loader = DataLoader(data, batch_size = self.train_params['batch_size'], shuffle = shuffle)
        elif type(data) == DataLoader:
            loader = data
        else:
            raise ValueError('Not accepted dataset type')
        return loader

    def train(self, dataloader):
        params = SimpleNamespace(**self.train_params)
        writer = SummaryWriter(params.log_dir)
        dataloader = self.get_dataloader(dataloader)

        self.model.to(device=self.device)
        self.model.train()
        for e in range(params.num_epochs):
            for t, (sample, label) in enumerate(dataloader):
                label = label.to(device=self.device, dtype=torch.long)
                sample = sample.to(device=self.device, dtype=torch.float32) # torch model weights is float32 by default. float64 would not work
                logit = self.forward(sample)
                
                params.optimizer.zero_grad()
                loss =  F.cross_entropy(logit, label)
                loss.backward()
                params.optimizer.step()

                if t % params.print_every == 0:
                    writer.add_scalar("Loss/train", loss.item(), e*len(dataloader)+t)
                    print('Epoch %d, Iteration %d, loss = %.4f' % (e, t, loss.item()))

                del label
                del logit
                del loss

            # Save model every print_every epochs
            if e > 0 and e % params.print_every == 0:
                torch.save(self.model.state_dict(), f"{params.checkpoint_path}/epoch_{e}")

    def forward(self, x):
        if self.model.task == "CPC":
            # x: N x 3 (context, future, negative) x samples x R x G x B
            embeds = []
            for n in range(x.shape[0]): # for each batch sample
                tup = []
                for samples in x[n]:
                    tup.append([self.encoder(sample) for sample in samples])
                embeds.append(tup)
            
            # embeds: N x 3 x samples x 4096 (samples are different between context, future, and negative
            context = self.modelgAR(embeds[:,0,:,:]) # N x 100
            
            z = [(context[n], embeds[n,1,:,:], embeds[n,2,:,:]) for n in range(len(embeds))]
        else:
            # If task == RP, embeds is a list/tuple of two embeddings
            #    task == TS, embeds is a list/tuple of three embeddings

            # indexing keeping dimension: https://discuss.pytorch.org/t/solved-simple-question-about-keep-dim-when-slicing-the-tensor/9280
            embeds = [self.model.encode(x[:,i:i+1]) for i in range(x.shape[1])] # x: N (Batch_size) x Sample_size x F
            if self.model.task == "RP":
                g = torch.abs(embeds[0] - embeds[1])
            elif self.model.task == "TS":
                g = torch.cat([torch.abs(embeds[0] - embeds[1]), torch.abs(embeds[1] - embeds[2])], dim=1)
            z = self.model.classify(g)
        
            del g
            del embeds
        return z
    
