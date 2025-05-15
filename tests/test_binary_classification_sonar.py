import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import pytest
import pandas as pd
from libs.ssl_task import Classification
from tests.test_utils import SonarDataModule
import lightning as L
from lightning.pytorch.loggers import CSVLogger
import shutil

@pytest.fixture
def setup_datamodule():
    L.seed_everything(7)
    """Fixture to set up the SonarDataModule."""
    datamodule = SonarDataModule(batch_size=16)
    datamodule.prepare_data()
    datamodule.setup('fit')
    return datamodule

def test_datamodule_initialization(setup_datamodule):
    """Test if the datamodule initializes correctly."""
    datamodule = setup_datamodule
    assert datamodule is not None

def test_training_process(setup_datamodule):
    """Test the training process."""
    datamodule = setup_datamodule

    # Initialize the model
    lit = Classification.ClassificationLit(
        encoder_path='tests.test_utils.Wide',
        encoder_kwargs={},
        task='binary',
        num_classes=2,
        optimizer_path='torch.optim.Adam',
        optimizer_kwargs={'lr': 0.002, 'weight_decay': 0.001},
    )

    # Set up the trainer
    logger = CSVLogger(save_dir='.')
    trainer = L.Trainer(
        max_epochs=50,
        max_steps=-1,
        logger=logger,
        log_every_n_steps=10,
    )

    # Let the training process run to completion
    trainer.fit(lit, datamodule)

    # Check if the metrics file is created
    metrics_file = Path(trainer.logger.log_dir) / "metrics.csv"
    assert metrics_file.exists()
    metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")
    for col in metrics.columns:
        if col.startswith("grad_2.0_norm") or "step" in col:
            del metrics[col]
    metrics.set_index("epoch", inplace=True)
    last_epoch_train = metrics.iloc[-1]
    assert last_epoch_train['train_accuracy_epoch'] > 0.9
    assert last_epoch_train['train_f1_epoch'] > 0.9
    assert last_epoch_train['train_loss_epoch'] < 0.25

    last_epoch_val = metrics.iloc[-2]
    assert last_epoch_val['val_Classifier/accuracy'] > 0.87
    assert last_epoch_val['val_Classifier/f1'] > 0.79
    assert last_epoch_val['val_Classifier/loss'] < 0.35

    # Clean up the logger directory
    shutil.rmtree(trainer.logger.log_dir)
