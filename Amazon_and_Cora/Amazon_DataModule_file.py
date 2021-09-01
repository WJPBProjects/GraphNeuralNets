import torch

from torch_geometric.datasets import Amazon
from torch_geometric.data import DataLoader as PyGDataLoader

import pytorch_lightning as pl
from typing import Optional
from torch.utils.data import random_split



class AmazonDataModule(pl.LightningDataModule):
"""
Class to hold Amazon graph data, either Photo or Products
"""
    def __init__(self, data_dir: str = "path/to/dir", batch_size: int = 32, transform = None, 
                num_workers = 8, mode = "Computers"):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transform
        self.num_workers = num_workers
        self.mode = mode
        
    def prepare_data(self):
        # download
        if self.mode == "Products":
            AmazonProducts(self.data_dir)
        elif self.mode == "Photo":
                           Amazon(self.data_dir, self.mode)
        else:
            Amazon(self.data_dir, self.mode)
        

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        self._full = Amazon(self.data_dir, self.mode, transform=self.transform)

    def return_dataloader(self):
        return PyGDataLoader(self._full, batch_size=self.batch_size, num_workers = self.num_workers)
    
    def train_dataloader(self):
        return self.return_dataloader()

    def val_dataloader(self):
        return self.return_dataloader()

    def test_dataloader(self):
        return self.return_dataloader()