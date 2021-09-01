"""
Architectures designed for UKBB, included here for the purpose of loading pretrained models and adapting them
"""
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
from torch_geometric.nn import GCNConv, GraphConv, SplineConv

from torch_geometric.nn.norm.batch_norm import BatchNorm

from torch.nn import ModuleList as ModuleList


class UKBB_Graph_Net(pl.LightningModule):
    def __init__(self, hidden_channels = 64, num_node_features = 3, num_classes = 2,
                     loss = None, measure = None, mode = 'classification',
                conv_layer = GraphConv, is_super = False):
        super(UKBB_Graph_Net, self).__init__()
        
        # If not being used as a base clase for something else
        if not is_super:
            self.conv1 = conv_layer(num_node_features, hidden_channels)
            self.conv2 = conv_layer(hidden_channels, hidden_channels)
            self.conv3 = conv_layer(hidden_channels, hidden_channels)
            self.lin = Linear(hidden_channels, num_classes)
            self.bnorm1 = BatchNorm(in_channels = hidden_channels)
            self.bnorm2 = BatchNorm(in_channels = hidden_channels)
            
        
        self.mode = mode
        
        #Set loss and measure to defaults
        if not loss:
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            self.criterion = loss
        
        if not measure:
            self.measure = accuracy
        else:
            self.measure = measure


    def forward(self, batch):
        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch
        
        if hasattr(batch, 'edge_weight'):
            edge_weight = batch.edge_weight
        else:
            edge_weight = None
        
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index, edge_weight)
        x = self.bnorm1(x)
        x = x.relu()
        
        x = self.conv2(x, edge_index, edge_weight)
        x = self.bnorm2(x)
        x = x.relu()
        
        x = self.conv3(x, edge_index, edge_weight)

        # 2. Readout layer
        x = global_mean_pool(x, batch_idx)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        return optimizer

    
    def training_step(self, train_batch, batch_idx):
        #Forward pass
        out = self.forward(train_batch)
        
        #calculate loss and acc and loga
        loss, acc = self.compute_metrics(out, train_batch.y)
        
        #log
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    
    def validation_step(self, val_batch, batch_idx):
        #Forward pass
        out = self.forward(val_batch)
        
        #calculate loss and acc and log
        loss, acc = self.compute_metrics(out, val_batch.y)
        
        #log
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        return loss
    
    
    def test_step(self, test_batch, batch_idx):        
        #Forward pass
        out = self.forward(test_batch)
        
        #calculate loss and acc and log
        loss, acc = self.compute_metrics(out, test_batch.y)
        
        #log
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        return loss

    
    def compute_metrics(self, model_output, targets):
        if self.mode == 'classification':
            loss = self.criterion(model_output, targets)
            acc = self.measure(model_output.argmax(dim=1), targets)
        
        elif self.mode == 'regression':    
            model_output = model_output.squeeze()
            targets = targets.squeeze()
            
            loss = self.criterion(model_output, targets)
            
            out = model_output.cpu().detach().numpy()
            targ = targets.cpu().detach().numpy()            
            acc = self.measure(out, targ)
        return loss, acc



class UKBB_Graph_Net_Spline(UKBB_Graph_Net):
    def __init__(self, hidden_channels = 64, num_node_features = 3, num_classes = 2,
                     loss = None, measure = None, mode = 'classification',
                conv_layer = SplineConv):
        super(UKBB_Graph_Net_Spline, self).__init__(loss = loss, measure = measure, mode = mode, 
                                            is_super = True)
        
        self.conv1 = SplineConv(num_node_features, hidden_channels, 
                                dim=3, kernel_size=5, aggr='add')
        self.conv2 = SplineConv(hidden_channels, hidden_channels,
                                dim=3, kernel_size=5, aggr='add')
        self.conv3 = SplineConv(hidden_channels, hidden_channels,
                                dim=3, kernel_size=5, aggr='add')
        self.lin = Linear(hidden_channels, num_classes)
        
        self.bnorm1 = BatchNorm(in_channels = hidden_channels)
        self.bnorm2 = BatchNorm(in_channels = hidden_channels)

    
    def forward(self, batch):
        x, edge_index, edge_attr, batch_idx = \
        batch.x, batch.edge_index, batch.edge_attr, batch.batch
        
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bnorm1(x)
        x = x.relu()
        
        x = self.conv2(x, edge_index, edge_attr)
        x = self.bnorm2(x)
        x = x.relu()
        
        x = self.conv3(x, edge_index, edge_attr)

        # 2. Readout layer
        x = global_mean_pool(x, batch_idx)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x


class UKBB_oversmoothing(UKBB_Graph_Net):
    def __init__(self, hidden_channels, num_node_features, num_classes, loss = None, 
                 measure = None, mode = 'classification', start_n_hops = 10, conv_layer = GraphConv, 
                 decrease_epochs = 1):
        
        super(UKBB_oversmoothing, self).__init__(loss = loss, measure = measure, mode = mode, 
                                            is_super = True)
        
        
        self.decrease_epochs = decrease_epochs
        
        #set input and output layers
        self.conv1 = conv_layer(num_node_features, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)
        
        self.conv_layer_list = []
        
        for _ in range(start_n_hops):
            new_layer = conv_layer(hidden_channels, hidden_channels)
            self.conv_layer_list.append(new_layer)
        
        self.conv_layer_list = ModuleList(self.conv_layer_list)
        self.n_hidden_layers = start_n_hops
    
    def forward(self, batch):
        x, edge_index, edge_attr, batch_idx = \
        batch.x, batch.edge_index, batch.edge_attr, batch.batch
        
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        
        for i, layer in enumerate(self.conv_layer_list):
            if i >= self.n_hidden_layers:
                break
            x = layer(x, edge_index)
            x = x.relu()

        # 2. Readout layer
        x = global_mean_pool(x, batch_idx)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x

    def on_train_epoch_end(self):
        if ((self.current_epoch+1)%self.decrease_epochs == 0) and self.n_hidden_layers>0:
            self.n_hidden_layers -= 1
            print(f"Now using {self.n_hidden_layers} hidden layers")




        
