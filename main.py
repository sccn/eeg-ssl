from libs.ssl_data import SSLHBNDataModule
from libs.ssl_utils import LitSSL
from lightning.pytorch.cli import LightningCLI

def main():
    cli = LightningCLI(LitSSL, SSLHBNDataModule, save_config_callback=None, subclass_mode_model=True)

if __name__ == '__main__':
    main()
