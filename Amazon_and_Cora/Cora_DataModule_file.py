import torch

from torch_geometric.datasets import CitationFull
from torch_geometric.data import DataLoader as PyGDataLoader

import pytorch_lightning as pl
from typing import Optional
from torch.utils.data import random_split



class CoraDataModule(pl.LightningDataModule):
    """
    Class to hold Cora citation network graph
    """

    def __init__(self, data_dir: str = "./Cora_data/", batch_size: int = 1, transform = None, 
                num_workers = 8, mode = "Cora"):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transform
        self.num_workers = num_workers
        self.mode = mode
        
    def prepare_data(self):
        # download
        CitationFull(self.data_dir, self.mode)
        

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        self._full = CitationFull(self.data_dir, self.mode, transform=self.transform)

    def return_dataloader(self):
        return PyGDataLoader(self._full, batch_size=self.batch_size, num_workers = self.num_workers)
    
    def train_dataloader(self):
        return self.return_dataloader()

    def val_dataloader(self):
        return self.return_dataloader()

    def test_dataloader(self):
        return self.return_dataloader()