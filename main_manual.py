from libs.ssl_data import SSLHBNDataModule
from libs.ssl_utils import LitSSL
from libs.ssl_task import RelativePositioning
from libs.ssl_model import VGGSSL
from lightning.pytorch.cli import LightningCLI
import lightning as L
from lightning.pytorch.loggers import WandbLogger

if __name__ == '__main__':
    model = VGGSSL()
    ssl_task = RelativePositioning(tau_pos_s=10, tau_neg_s=None, n_samples_per_dataset=1)
    lit = RelativePositioning.RelativePositioningLit(encoder=model, encoder_emb_size=1024, emb_size=100, dropout=0.5)
    dataloader = SSLHBNDataModule(ssl_task=ssl_task, target_label='p_factor')
    wandb_logger = False #WandbLogger(project="eeg-ssl")
    trainer = L.Trainer(max_epochs=2, use_distributed_sampler=True, check_val_every_n_epoch=1, logger=wandb_logger)
    trainer.fit(lit, dataloader)

