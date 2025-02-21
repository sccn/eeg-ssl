import torch
from torch import optim
from torch.nn import functional as F
import torch.nn as nn
from .evaluation import train_regressor, RankMe
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
        self.emb_size = emb_size
        encoder_expected_emb_size = 1024
        if encoder_emb_size != encoder_expected_emb_size:
            projection_layer = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(encoder_emb_size, encoder_expected_emb_size),
            )
        else:
            projection_layer = nn.Identity()
            
        self.embedder = nn.Sequential(
            projection_layer,
            nn.Dropout(dropout),
            nn.Linear(encoder_expected_emb_size, emb_size),
            nn.Dropout(dropout)
        )
            
        evaluators = ['RankMe']
        self.evaluators = [globals()[evaluator]() for evaluator in evaluators]
        
    def embed(self, x):
        return self.embedder(self.encoder(x))

    def training_step(self, batch, batch_idx):
        raise NotImplementedError()

    def validation_step(self, batch, batch_idx):
        X, Y, _ = batch
        z = self.embed(X)
        for evaluator in self.evaluators:
            evaluator.update(z)
            # self.log('valid_acc', evaluator, on_step=False, on_epoch=True)
        # pass
        
    def test_step(self, batch, batch_idx):
        # this is the test loop
        X, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        test_loss = F.mse_loss(x_hat, x)
        self.log("test_loss", test_loss)

    def on_validation_epoch_end(self):
    #     # log epoch metric
        for evaluator in self.evaluators:
            # self.log(f'val_{type(evaluator).__name__}', evaluator, on_step=False, on_epoch=True)
            self.log(f'val_{type(evaluator).__name__}', evaluator.compute())
            evaluator.reset()
    #     pass
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
