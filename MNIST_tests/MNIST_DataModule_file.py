import torch

from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.data import DataLoader as PyGDataLoader

import pytorch_lightning as pl
from typing import Optional
from torch.utils.data import random_split



class MNISTDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str = "path/to/dir", batch_size: int = 32, transform = None, 
                num_workers = 8):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transform
        self.num_workers = num_workers
        
    def prepare_data(self):
        # download
        MNISTSuperpixels(self.data_dir, train=True)
        MNISTSuperpixels(self.data_dir, train=False)

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            mnist_full = MNISTSuperpixels(self.data_dir, train=True,
                               transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

            # Optionally...
            # self.dims = tuple(self.mnist_train[0][0].shape)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.mnist_test = MNISTSuperpixels(self.data_dir, train=False, transform=self.transform)

            # Optionally...
            # self.dims = tuple(self.mnist_test[0][0].shape)

    def train_dataloader(self):
        return PyGDataLoader(self.mnist_train, batch_size=self.batch_size, num_workers = self.num_workers)

    def val_dataloader(self):
        return PyGDataLoader(self.mnist_val, batch_size=self.batch_size, num_workers = self.num_workers)

    def test_dataloader(self):
        return PyGDataLoader(self.mnist_test, batch_size=self.batch_size, num_workers = self.num_workers)