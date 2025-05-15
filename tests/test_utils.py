import lightning as L
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
import os
from pathlib import Path

BATCH_SIZE = 256 if torch.cuda.is_available() else 64
NUM_WORKERS = int(os.cpu_count() / 2)
class CIFARDataModule(L.LightningDataModule):
    def __init__(self,
                batch_size: int = BATCH_SIZE,
                num_workers: int = NUM_WORKERS,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def prepare_data(self):
        torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=self.transform)
        torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=self.transform)

    def get_splits(self, len_dataset, val_split):
        """Computes split lengths for train and validation set."""
        if isinstance(val_split, int):
            train_len = len_dataset - val_split
            splits = [train_len, val_split]
        elif isinstance(val_split, float):
            val_len = int(val_split * len_dataset)
            train_len = len_dataset - val_len
            splits = [train_len, val_len]
        else:
            raise ValueError(f"Unsupported type {type(val_split)}")

        return splits

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                          download=False, transform=self.transform)
            len_dataset = len(self.train_set)
            splits = self.get_splits(len_dataset, 0.2)
            self.train_set, self.val_set = random_split(self.train_set, splits, generator=torch.Generator().manual_seed(42))

        if stage == 'test' or stage is None:
            self.test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                         download=False, transform=self.transform)
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size,
                                           shuffle=True, num_workers=self.num_workers)
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_set, batch_size=self.batch_size,
                                           shuffle=False, num_workers=self.num_workers)
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_set, batch_size=self.batch_size,
                                           shuffle=False, num_workers=self.num_workers)


class CIFARConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CIFARResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet18(num_classes=10)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.model.maxpool = nn.Identity()

    def forward(self, x):
        return self.model(x)

import pandas as pd
import os
import zipfile
import requests

def download_and_extract_sonar_data():
    # Define the URL and target paths
    url = "https://www.kaggle.com/api/v1/datasets/download/rupakroy/sonarcsv"
    download_path = Path(__file__).resolve().parent / "data" / "sonarcsv.zip" # os.path.expanduser("~/Downloads/sonarcsv.zip")
    extract_path = Path(__file__).resolve().parent / "data" 

    # Ensure the target directory exists
    os.makedirs(extract_path, exist_ok=True)

    # Download the zip file
    print("Downloading sonar dataset...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(download_path, "wb") as f:
            f.write(response.content)
        print(f"Downloaded to {download_path}")
    else:
        raise Exception(f"Failed to download file. Status code: {response.status_code}")

    # Extract the zip file
    print("Extracting sonar dataset...")
    with zipfile.ZipFile(download_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)
    print(f"Extracted to {extract_path}")

    # Clean up the downloaded zip file
    os.remove(download_path)
    print(f"Removed downloaded zip file: {download_path}")

class SonarDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        data = pd.read_csv(data_path, header=None)
        self.data = data.iloc[:, 0:60]
        self.labels = data.iloc[:, 60].map({'M': 0, 'R': 1})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data.iloc[idx].values, dtype=torch.float32)
        y = torch.tensor(self.labels.iloc[idx], dtype=torch.long)
        return x, y

class SonarDataModule(L.LightningDataModule):
    def __init__(self,
                batch_size: int = BATCH_SIZE,
                num_workers: int = NUM_WORKERS,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_path = Path(__file__).resolve().parent / "data" / "sonar.csv"

    def prepare_data(self):
        # Download and extract the dataset if not already present
        if not os.path.exists(self.data_path):
            download_and_extract_sonar_data()

    def get_splits(self, len_dataset, val_split):
        """Computes split lengths for train and validation set."""
        if isinstance(val_split, int):
            train_len = len_dataset - val_split
            splits = [train_len, val_split]
        elif isinstance(val_split, float):
            val_len = int(val_split * len_dataset)
            train_len = len_dataset - val_len
            splits = [train_len, val_len]
        else:
            raise ValueError(f"Unsupported type {type(val_split)}")

        return splits

    def setup(self, stage=None):
        data = SonarDataset(self.data_path)
        len_dataset = len(data)
        splits = self.get_splits(len_dataset, 0.2)
        self.train_set, self.val_set = random_split(data, splits, generator=torch.Generator().manual_seed(42))
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size,
                                           shuffle=True, num_workers=self.num_workers)
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_set, batch_size=self.batch_size,
                                           shuffle=False, num_workers=self.num_workers)

class Wide(nn.Module):
    # instantiate all modules (their attributes and methods)
    def __init__(self):
        # initialize attributes and methods of the parent class
        super().__init__()
        # input layer for 60 variables (60 units or neurons) and 180 output units
        self.hidden = nn.Linear(in_features=60, out_features=180)
        # activation of the layer (breaking linearity)
        self.relu = nn.ReLU()
        # the output is a real number for binary classification...
        self.output = nn.Linear(in_features=180, out_features=2)
        # ...and the sigmoid takes the input (1) tensor and squeeze (reescale) it to [0,1] range
        # representing the probability of the target label of a given sample.
        # class 1 = P, class 2 = 1 - P(class 1)
        # Note: sigmoid is used for binary classification, softmax is an extension of sigmoid for multiclass problems
        # self.sigmoid = nn.Sigmoid()

    # the forward function defines the neural network structure, with
    # number of units (neurons), activations, regularizations, outputs...
    # Then, here we define how the network will be run from input to output:
    def forward(self, x):
        # taking the input, computing weigths and applying non-linearity
        x = self.output(self.relu(self.hidden(x)))
        # taking the output of the previous layer and squeezing it
        # to the range [0,1]
        # x = self.sigmoid(self.output(x))
        return x  # x is the probability of class 1, while class 2 is (1-x)

