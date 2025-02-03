from libs.ssl_data import RelativePositioningHBNDataModule
from libs.ssl_model import LitSSL
from lightning.pytorch.cli import LightningCLI
import os


def main():
    cli = LightningCLI(LitSSL, RelativePositioningHBNDataModule, save_config_callback=None)

if __name__ == '__main__':
    main()
