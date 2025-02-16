from libs.ssl_data import RelativePositioningHBNDataModule
from libs.ssl_model import LitSSL, VGGSSL
from lightning.pytorch.cli import LightningCLI

def main():
    cli = LightningCLI(LitSSL, RelativePositioningHBNDataModule, save_config_callback=None, subclass_mode_model=True)

if __name__ == '__main__':
    main()
