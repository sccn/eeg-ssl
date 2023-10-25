import torch
import torchvision.models as torchmodels
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from types import SimpleNamespace
from torch.utils.tensorboard import SummaryWriter
import os

class SSLModel(ABC):
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

class VGGSSL(SSLModel, nn.Module):
    def __init__(self, model_params=None):
        super().__init__(model_params)
        self.encoder = torchmodels.vgg16(weights=self.weights)

        self.__model_augment()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def __model_augment(self):
        self.encoder.classifier = torch.nn.Sequential(*list(self.encoder.classifier.children())[:-3])
        if self.task == "RP":
            self.classifier = nn.Linear(4096, 2)
        elif self.task == "TS":
            self.classifier = nn.Linear(4096*2, 2)
        elif self.task == "CPC":
            self.gAR = nn.GRU(4096, 100) # hidden size = 100, per (Banville et al, 2020) experiment

    def forward(self, x):
        if self.task == "CPC":
            # x: N x 3 (context, future, negative) x samples x R x G x B
            embeds = []
            for n in range(x.shape[0]): # for each batch sample
                tup = []
                for samples in x[n]:
                    tup.append([self.encoder(sample) for sample in samples])
                embeds.append(tup)
            
            # embeds: N x 3 x samples x 4096 (samples are different between context, future, and negative
            context = self.gAR(embeds[:,0,:,:]) # N x 100
            
            z = [(context[n], embeds[n,1,:,:], embeds[n,2,:,:]) for n in range(len(embeds))]
        else:
            # If task == RP, embeds is a list/tuple of two embeddings
            #    task == TS, embeds is a list/tuple of three embeddings
            embeds = [self.encoder(x[:,i,:,:,:]) for i in range(x.shape[1])] # x: N (Batch_size) x Sample_size x R x G x B
            if self.task == "RP":
                g = torch.abs(embeds[0] - embeds[1])
            elif self.task == "TS":
                g = torch.cat([torch.abs(embeds[0] - embeds[1]), torch.abs(embeds[1] - embeds[2])], dim=1)
            z = self.classifier(g)
        
            del g
            del embeds
        return z
    
class SSLModelUtils():
    def __init__(self,
            model_params={
                'model': 'VGGSSL'
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

        self.model = globals()[model_params['model']](model_params)
        self.train_params = train_params 
        self.__init_train()
    
    def __init_train(self):
        '''
        Initialize training parameters
        '''
        if not os.path.exists(self.train_params['checkpoint_path']):
            os.mkdir(self.train_params['checkpoint_path'])
        if not os.path.exists(self.train_params['log_dir']):
            os.mkdir(self.train_params['log_dir'])
        if not self.train_params['optimizer']:
            self.train_params['optimizer'] = torch.optim.Adamax(self.model.parameters(), lr = self.train_params['learning_rate'])

    def train(self, dataloader):
        params = SimpleNamespace(**self.train_params)
        writer = SummaryWriter(params.log_dir)

        self.model.train()
        for e in range(params.num_epochs):
            for t, (sample, label) in enumerate(dataloader):
                label = label.to(device=self.device, dtype=torch.long)
                sample = sample.to(device=self.device)
                logit = self(sample)
                
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
                torch.save(self.state_dict(), f"{params.checkpoint_path}/epoch_{e}")

    
