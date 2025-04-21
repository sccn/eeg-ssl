import torch
from torch import optim
from torch.nn import functional as F
import torch.nn as nn
from .evaluation import RankMe, Regressor
import lightning as L

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
        # self.emb_size = emb_size
        # encoder_expected_emb_size = 1024
        # if encoder_emb_size != encoder_expected_emb_size:
        #     projection_layer = nn.Sequential(
        #         nn.Dropout(dropout),
        #         nn.Linear(encoder_emb_size, encoder_expected_emb_size),
        #     )
        # else:
        #     projection_layer = nn.Identity()
            
        # self.embedder = nn.Sequential(
        #     projection_layer,
        #     nn.Dropout(dropout),
        #     nn.Linear(encoder_expected_emb_size, emb_size),
        #     nn.Dropout(dropout)
        # )
            
        evaluators = ['RankMe', 'Regressor']
        self.evaluators = [globals()[evaluator]() for evaluator in evaluators]
        
    def embed(self, x):
        return self.embedder(self.encoder(x))

    def on_train_start(self):
        self.train()

    def training_step(self, batch, batch_idx):
        raise NotImplementedError()
    
    def on_validation_start(self):
        # print(self.training)
        # print(self.eval)
        self.eval()

    def validation_step(self, batch, batch_idx):
        X, Y, _, subjects = batch
        z = self.embed(X)

        for evaluator in self.evaluators:
            evaluator.update((z, Y, subjects))
        
    def test_step(self, batch, batch_idx):
        X, Y, _, subjects = batch
        z = self.embed(X)

        for evaluator in self.evaluators:
            evaluator.update((z, Y, subjects))

    def predict_step(self, batch, batch_idx):
        X, Y, _ = batch
        z = self.embed(X)

        return z

    def on_validation_epoch_end(self):
        for evaluator in self.evaluators:
            val =  evaluator.compute()
            if type(val) == dict:
                for k, v in val.items():
                    self.log(f'val_{type(evaluator).__name__}/{k}', v)
                    print(f'val_{type(evaluator).__name__}/{k}', v)
            else:
                self.log(f'val_{type(evaluator).__name__}', val)
                print(f'val_{type(evaluator).__name__}', val)
    
    def on_test_epoch_end(self):
        for evaluator in self.evaluators:
            val =  evaluator.compute()
            if type(val) == dict:
                for k, v in val.items():
                    self.log(f'val_{type(evaluator).__name__}/{k}', v)
                    print(f'val_{type(evaluator).__name__}/{k}', v)
            else:
                self.log(f'val_{type(evaluator).__name__}', val)
                print(f'val_{type(evaluator).__name__}', val)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        return optimizer
