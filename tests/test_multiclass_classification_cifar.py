import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pytest
import pandas as pd
from libs.ssl_task import Classification
from tests.test_utils import CIFARDataModule
import lightning as L
from lightning.pytorch.loggers import CSVLogger
import shutil

@pytest.fixture
def setup_cifar_datamodule():
    """Fixture to set up the CIFARDataModule."""
    L.seed_everything(7)
    datamodule = CIFARDataModule(batch_size=256)
    datamodule.prepare_data()
    datamodule.setup('fit')
    return datamodule

def test_cifar_datamodule_initialization(setup_cifar_datamodule):
    """Test if the CIFAR datamodule initializes correctly."""
    datamodule = setup_cifar_datamodule
    assert datamodule is not None

def test_cifar_training_process(setup_cifar_datamodule):
    """Test the training process for CIFAR."""
    datamodule = setup_cifar_datamodule

    # Initialize the model
    lit = Classification.ClassificationLit(
        encoder_path='tests.test_utils.CIFARResNet',
        encoder_kwargs={},
        task='multiclass',
        num_classes=10,
        optimizer_path='torch.optim.Adamax',
        optimizer_kwargs={'lr': 0.005, 'weight_decay': 0.001},
    )

    # Set up the trainer
    logger = CSVLogger(save_dir='.')
    trainer = L.Trainer(
        max_epochs=5,
        max_steps=-1,
        logger=logger,
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
    assert last_epoch_train['train_accuracy_epoch'] > 0.7
    assert last_epoch_train['train_loss_epoch'] < 0.8

    last_epoch_val = metrics.iloc[-2]
    assert last_epoch_val['val_Classifier/accuracy'] > 0.6
    assert last_epoch_val['val_Classifier/loss'] < 1

    # Clean up the logger directory
    shutil.rmtree(trainer.logger.log_dir)