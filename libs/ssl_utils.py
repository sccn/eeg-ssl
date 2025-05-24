from typing import Any, Optional, Union
import torch
from torch import optim
from .evaluation import RankMe, Regressor
import lightning as L
import importlib
import pathlib
from lightning.pytorch.utilities import grad_norm

class LitSSL(L.LightningModule):
    def __init__(self, 
        encoder_path: str,
        encoder_kwargs: Optional[Union[dict[str, Any], dict[str, dict[str, Any]]]] = None,
        emb_size=100, 
        encoder_emb_size=100,
        learning_rate=0.005,
        seed=0,
        optimizer_path = None,
        optimizer_kwargs = None,
        window_norm='None',
    ):
        super().__init__()
        self.encoder = instantiate_module(encoder_path, encoder_kwargs)
        self.encoder.train()
        self.emb_size = emb_size
        self.encoder_emb_size = encoder_emb_size
        self.learning_rate = learning_rate

        evaluators = ['RankMe', 'Regressor']
        self.evaluators = [globals()[evaluator]() for evaluator in evaluators]

        self.save_hyperparameters()

    def embed(self, x):
        pass

    def remove_chan(self, x):
        """
        Remove Cz channel from the input tensor.
        Assumes that the Cz channel is last channel
        """
        assert x.ndim == 3, f"Input tensor must be 3D, got {x.ndim}D tensor"
        assert x.shape[1] == 129 or x.shape[1] == 128, f"Input tensor must have 129 or 128 channels, got {x.shape[1]} channels"
        if x.shape[1] == 129:
            # check that Cz is empty
            assert torch.all(x[:, -1, :] == 0), f"Cz channel is not empty, found {x[..., -1].unique()}"
            # remove Cz channel - last channel
            x = x[:, :-1, :]
        return x
    
    def normalize_data(self, x, method='robust'):
        if self.hparams.window_norm == 'None':
            # no normalization
            return x
        elif self.hparams.window_norm == 'all':
            channel_wise_norm = False
        elif self.hparams.window_norm == 'channel_wise':
            channel_wise_norm = True
        if method == 'standard':
            if channel_wise_norm:
                # normalize each channel separately
                center = x.mean(dim=-1, keepdim=True)
                variance = x.std(dim=-1, keepdim=True) #+ 1e-7  # add small epsilon for numerical stability
            else:
                center = x.mean(dim=0, keepdim=True)  
                variance = x.std(dim=0, keepdim=True) #+ 1e-7  # add small epsilon for numerical stability
        elif method == 'robust':
            if channel_wise_norm:
                # normalize each channel separately
                center, _ = x.median(dim=-1, keepdim=True)
                variance = x.quantile(0.75, dim=-1, keepdim=True) - x.quantile(0.25, dim=-1, keepdim=True)
            else:
                center, _ = x.median(dim=0, keepdim=True)  
                variance = x.quantile(0.75, dim=0, keepdim=True) - x.quantile(0.25, dim=0, keepdim=True)
        # standard scale to 0 mean and 1 std using statistics of the entire recording
        x = (x - center) / variance # normalize preserving batch dim
        return x
        # mean = x.mean(dim=-1, keepdim=True)
        # std = x.std(dim=-1, keepdim=True) + 1e-7  # add small epsilon for numerical stability
        # x = (x - mean) / std
        # return x

    def training_step(self, batch, batch_idx):
        raise NotImplementedError()
    
    def validation_step(self, batch, batch_idx):
        X, Y, _, subjects = batch
        z = self.embed(X)

        for evaluator in self.evaluators:
            evaluator.update((z, Y, subjects))
        
    def on_test_start(self):
        for evaluator in self.evaluators:
            assert len(evaluator.labels) == 0, f"Evaluator {type(evaluator).__name__} should be empty at the start of validation"

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
            else:
                self.log(f'val_{type(evaluator).__name__}', val)
            
    def on_test_epoch_end(self):
        for evaluator in self.evaluators:
            val =  evaluator.compute()
            if type(val) == dict:
                for k, v in val.items():
                    self.log(f'test_{type(evaluator).__name__}/{k}', v)
            else:
                self.log(f'test_{type(evaluator).__name__}', val)

    def on_keyboard_interrupt(self):
        self.trainer.save_checkpoint("last_checkpoint.ckpt")

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self, norm_type=2)
        self.log_dict(norms)
        
    def configure_optimizers(self):
        if self.hparams.optimizer_path:
            optimizer = instantiate_module(self.hparams.optimizer_path, self.hparams.optimizer_kwargs, args=[self.parameters()])
        else:
            print('learning rate', self.learning_rate)
            optimizer = optim.Adamax(self.parameters(), lr=self.learning_rate)
        print('optimizer', optimizer)
        return optimizer
    
############### Helper functions ###############
def instantiate_module(module_class_str, kwargs, args=None):
    """
    Instantiates a class from a module and class name string,
    passing constructor arguments as a dictionary.
    If any argument contains a string resembling a classpath, extract it (but do not import).

    Args:
      module_class_str: A string in the format "module_name.ClassName".
      kwargs: A dictionary of keyword arguments for the class constructor.

    Returns:
      A tuple: (instance of the class, list of extracted classpaths as strings).
      Returns (None, None) if an error occurs.
    """
    try:
        module_name, class_name = module_class_str.rsplit('.', 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        # assert issubclass(cls, torch.nn.Module), f"{cls} is not a subclass of torch Module"

        # Check for classpaths in kwargs
        for key, value in kwargs.items():
            if isinstance(value, str) and is_potential_classpath(value):
              classpath = get_class_from_string(value)
              kwargs[key] = classpath

        if args is not None:
            instance = cls(*args, **kwargs)
        else:
            instance = cls(**kwargs)
        return instance
    except (ImportError, AttributeError, ValueError) as e:
        print(f"Error instantiating class: {e}")
        return None, None

def is_potential_classpath(value):
    """
    Basic check to see if string value might be a classpath (e.g., contains dot notation)
    and might be used to reference a class. 
    """
    # Check for typical classpath patterns. 
    # This is a basic check; enhance as needed.
    if isinstance(value, str) and "." in value:
        # Ensure the string is not an actual existing file path (use pathlib for robust check):
        if not pathlib.Path(value).exists():
          return True
    return False

def get_class_from_string(class_path: str):
    """
    Dynamically imports a class from a string representation of its path.

    Args:
        class_path: A string representing the full path to the class,
                    e.g., 'module.submodule.ClassName'.

    Returns:
        The class object if found, otherwise raises ImportError or AttributeError.
    """
    try:
        module_path, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except ImportError as e:
        raise ImportError(f"Could not import module '{module_path}': {e}")
    except AttributeError:
        raise AttributeError(f"Class '{class_name}' not found in module '{module_path}'")
