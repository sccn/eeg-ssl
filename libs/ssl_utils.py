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
            
        self.clf = nn.Linear(emb_size, 1)
        
        self.rankme = RankMe()
        
    def embed(self, x):
        return self.embedder(self.encoder(x))

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        X, y = batch
        x1, x2 = X[0], X[1]
        z1, z2 = self.embed(x1), self.embed(x2)
        loss = self.loss(z1, z2, y, self.clf)
        # z = torch.abs(z1 - z2)
        # loss = nn.functional.binary_cross_entropy_with_logits(self.clf(z).flatten(), y)

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
