import os.path as osp
import numpy as np

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import pytorch_lightning as pl

from argparse import ArgumentParser
from torch.nn import Linear
from torch_geometric.datasets import FAUST
from torch_geometric.data import DataLoader
from torch_geometric.nn import SplineConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import accuracy
from torch_geometric.nn import GCNConv, GraphConv, SplineConv, avg_pool_neighbor_x

from torch.nn import ModuleList as ModuleList


class ukbb_oversmooth_net_mask(pl.LightningModule):
    """
    Model adapted from UKBB for node classificatino tasks and oversmoothing
    """
    
    def __init__(self, hidden_channels = 64, num_node_features = 3, num_classes = 2,
                 CBS_initial_neighb_distance = 0, CBS_epochs = 1, starting_own_weight = 0.5,
                 weight_epochs = None, weight_incr = None):
        super(ukbb_oversmooth_net_mask, self).__init__()
        
        # If not being used as a base clase for something else
        self.conv1 = GraphConv(num_node_features, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)
        
        self.avg_pool = avg_pool_neighbor_x
        
        self.criterion = torch.nn.CrossEntropyLoss()
        self.measure = accuracy
        
        self.CBS = CBS_initial_neighb_distance
        self.CBS_epochs = CBS_epochs
        
        self.own_weight = starting_own_weight
        self.weight_epochs = weight_epochs
        self.weight_incr = weight_incr        
       
       #set mode  
        if self.CBS > 0 or self.own_weight!=1:
            self.CBS_on = True
        else:
            self.CBS_on = False
        
        print(f"CBS mode on: {self.CBS_on}")

    def forward(self, batch):
        edge_index, batch_idx = batch.edge_index, batch.batch
        
        # 1. Obtain node embeddings
        batch.x = self.conv1(batch.x, edge_index)
        if self.training and self.CBS_on:
            self.smooth_data_x2(batch)
        batch.x = batch.x.relu()
        
        batch.x = self.conv2(batch.x, edge_index)
        if self.training and self.CBS_on:
            self.smooth_data_x2(batch)
        batch.x = batch.x.relu()
        
        x = self.conv3(batch.x, edge_index)
        
        x = self.lin(x)
        x = x.softmax(dim = 1)
        return x

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        return optimizer

    
    def training_step(self, train_batch, batch_idx):
        mask = train_batch.train_mask
        
        #Forward pass
        out = self.forward(train_batch)

        #calculate loss and acc and loga
        loss, acc = self.compute_metrics(out[mask], train_batch.y[mask])
        
        #log
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    
    def validation_step(self, val_batch, batch_idx):
        mask = val_batch.val_mask
        
        #Forward pass
        out = self.forward(val_batch)
        
        #calculate loss and acc and log
        loss, acc = self.compute_metrics(out[mask], val_batch.y[mask])
        
        #log
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        return loss
    
    
    def test_step(self, test_batch, batch_idx):       
        mask = test_batch.test_mask
        
        #Forward pass
        out = self.forward(test_batch)
        
        #calculate loss and acc and log
        loss, acc = self.compute_metrics(out[mask], test_batch.y[mask])
        
        #log
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        return loss

    
    def compute_metrics(self, model_output, targets):
        loss = self.criterion(model_output, targets)
        preds = model_output.argmax(dim=1)
        acc = self.measure(preds, targets)
        return loss, acc


    def on_train_epoch_end(self):    
        if self.CBS > 0:
            if ((self.current_epoch+1) % self.CBS_epochs) == 0:
                self.CBS -= 1
                print(f"now using CBS loop of {self.CBS}")
        
        if self.own_weight < 1:
            if ((self.current_epoch+1) % self.weight_epochs) == 0:
                self.own_weight += self.weight_incr
                if self.own_weight >1:
                    self.own_weight = 1
                print(f"now using self weight of {self.own_weight}")

    def smooth_data_x2(self, data):
        x_tmp = data.x
        for i in range(self.CBS):
            self.avg_pool(data)
        data.x = (self.own_weight * x_tmp) + ((1-self.own_weight)*data.x)