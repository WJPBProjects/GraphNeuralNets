"""
A variety of network architectures to try for the purpose of OASIS AD classification. Inspired by
Work on UKBB. Recommendation is to use models with "Binary" suffix, as these were ultimately used
and are most refined (UKBB_All_Substructs_Net2_Binary, Oasis_PCA_Net_Binary)
"""

import os.path as osp
import numpy as np

import sklearn.decomposition as decomp
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import pytorch_lightning as pl

from torch.nn import Linear, Sigmoid, Sequential, ReLU
from torch.nn import Dropout as LinearDropout



from argparse import ArgumentParser
from torch_geometric.datasets import FAUST
from torch_geometric.data import DataLoader
from torch_geometric.nn import SplineConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import accuracy
from torch_geometric.nn import GCNConv, GraphConv, SplineConv
from torch_geometric.nn.norm.batch_norm import BatchNorm

from torch.nn import ModuleList as ModuleList


        
        
###########NEXT MODEL################################################################################################################################################################################################################################################




class UKBB_All_Substructs_Net(pl.LightningModule):
    def __init__(self, hidden_channels = 64, num_node_features = 3, 
                 num_classes = 2, conv_layer = GraphConv, num_sub_structs = 15):
        super(UKBB_All_Substructs_Net, self).__init__()
        
        self.n_sub_structs = num_sub_structs
        self.num_classes = num_classes
        
        self.conv1 = conv_layer(num_node_features, hidden_channels)
        self.conv2 = conv_layer(hidden_channels, hidden_channels)
        self.conv3 = conv_layer(hidden_channels, hidden_channels)
        
        self.lin1 = Linear(hidden_channels, 1)
        self.lin2 = Linear(num_sub_structs, num_classes)
        
        self.bnorm1 = BatchNorm(in_channels = hidden_channels)
        self.bnorm2 = BatchNorm(in_channels = hidden_channels)
        
        
        self.criterion = torch.nn.CrossEntropyLoss()
        self.measure = accuracy
        
        
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        return optimizer

    
    def training_step(self, batch, batch_idx):
        #Forward pass, #calculate loss and acc and loga
        num_graphs = batch["y"].shape[0]
        out = self.forward(batch["x"], num_graphs)
        loss, acc = self.compute_metrics(out, batch["y"])
        
        #log
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    
    def validation_step(self, batch, batch_idx):
        #Forward pass, #calculate loss and acc and loga
        num_graphs = batch["y"].shape[0]
        out = self.forward(batch["x"], num_graphs)
        loss, acc = self.compute_metrics(out, batch["y"])
        
        #log
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        return loss
    
    
    def test_step(self, batch, batch_idx):
        #Forward pass, #calculate loss and acc and loga
        num_graphs = batch["y"].shape[0]
        out = self.forward(batch["x"], num_graphs)
        loss, acc = self.compute_metrics(out, batch["y"])
        
        #log
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        return loss
    
    def compute_metrics(self, model_output, targets):
        loss = self.criterion(model_output, targets)
        acc = self.measure(model_output.argmax(dim = 1), targets)
        return loss, acc


    def forward(self, batch, num_graphs):
        embeddings = torch.empty((num_graphs,0)).to(self.device)
        for graph in batch:
            x = self.one_graph_forward(graph)
            embeddings = torch.hstack((embeddings, x))
        
        x = embeddings.squeeze().reshape(-1, self.n_sub_structs)
        x = self.lin2(x)
        return x
    
    def one_graph_forward(self, batch):
        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch
        
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = self.bnorm1(x)
        x = x.relu()
        
        x = self.conv2(x, edge_index)
        x = self.bnorm2(x)
        x = x.relu()
        
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch_idx)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin1(x)
        return x
        
        
        
###########NEXT MODEL################################################################################################################################################################################################################################################



from UKBB_Graph_Net_file import UKBB_Graph_Net, UKBB_Graph_Net_Spline
from torch.nn import ModuleList



class UKBB_All_Substructs_Net2(pl.LightningModule):
    def __init__(self, hidden_channels = 64, num_node_features = 3, 
                 num_classes = 2, conv_layer = GraphConv, num_sub_structs = 15, model_type = 'gcn'):
        super(UKBB_All_Substructs_Net2, self).__init__()
        
        self.n_sub_structs = num_sub_structs
        self.num_classes = num_classes
        
        
        assert (model_type == 'gcn' or model_type == 'spline')
        #NOTE HARDCODED 1 in the loop below

        # NEW CODE
        self.all_gnns = ModuleList()
        for i in range(num_sub_structs):
            if model_type == 'spline':
                new_gcn = UKBB_Graph_Net_Spline(
                    hidden_channels = hidden_channels, 
                    num_node_features = num_node_features,      
                    num_classes = 1
                )
            else:
                new_gcn = UKBB_Graph_Net(
                    hidden_channels = hidden_channels, 
                    num_node_features = num_node_features,      
                    num_classes = 1, 
                    conv_layer = conv_layer,
                )
            self.all_gnns.append(new_gcn)
        
        self.lin2 = Linear(num_sub_structs, num_classes)
        
        self.criterion = torch.nn.CrossEntropyLoss()
        self.measure = accuracy
        
        
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        return optimizer

    
    def training_step(self, batch, batch_idx):
        #Forward pass, #calculate loss and acc and loga
        num_graphs = batch["y"].shape[0]
        out = self.forward(batch["x"], num_graphs)
        loss, acc, unhealthy_acc = self.compute_metrics(out, batch["y"], unhealthy_seperate = True)
        
        #log
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        self.log('unhealthy_train_acc', unhealthy_acc)
        return loss

    
    def validation_step(self, batch, batch_idx):
        #Forward pass, #calculate loss and acc and loga
        num_graphs = batch["y"].shape[0]
        out = self.forward(batch["x"], num_graphs)
        loss, acc = self.compute_metrics(out, batch["y"])
        
        #log
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        return loss
    
    
    def test_step(self, batch, batch_idx):
        #Forward pass, #calculate loss and acc and loga
        num_graphs = batch["y"].shape[0]
        out = self.forward(batch["x"], num_graphs)
        loss, acc = self.compute_metrics(out, batch["y"])
        
        #log
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        return loss
    
    def compute_metrics(self, model_output, targets, unhealthy_seperate = False):
        loss = self.criterion(model_output, targets)
        acc = self.measure(model_output.argmax(dim = 1), targets)
        if unhealthy_seperate:
            mask = (targets == 1)
            if mask.sum() > 0:
                unhealthy_acc = self.measure(model_output[mask].argmax(dim = 1), targets[mask])
            else:
                unhealthy_acc = 0
            return loss, acc, unhealthy_acc
        
        return loss, acc


    def forward(self, batch, num_graphs):
        embeddings = torch.empty((num_graphs,0)).to(self.device)
        for i, graph in enumerate(batch):
            x = self.all_gnns[i](graph)
            embeddings = torch.hstack((embeddings, x))
        
        x = embeddings.squeeze().reshape(-1, self.n_sub_structs)
        x = self.lin2(x)
        return x

        
###########NEXT MODEL ################################################################################################################################################################################################################################################


class UKBB_All_Substructs_Net2_Binary(pl.LightningModule):
    def __init__(self, hidden_channels = 64, num_node_features = 3, conv_layer = GraphConv, num_sub_structs = 15,
                 model_type = 'gcn', threshold = 0.5):
        super(UKBB_All_Substructs_Net2_Binary, self).__init__()
        
        self.n_sub_structs = num_sub_structs
        
        assert (model_type == 'gcn' or model_type == 'spline')
        #NOTE HARDCODED 1 in the loop below

        # NEW CODE
        self.all_gnns = ModuleList()
        for i in range(num_sub_structs):
            if model_type == 'spline':
                new_gcn = UKBB_Graph_Net_Spline(
                    hidden_channels = hidden_channels, 
                    num_node_features = num_node_features,      
                    num_classes = 1
                )
            else:
                new_gcn = UKBB_Graph_Net(
                    hidden_channels = hidden_channels, 
                    num_node_features = num_node_features,      
                    num_classes = 1, 
                    conv_layer = conv_layer,
                )
            self.all_gnns.append(new_gcn)
        
        self.criterion = torch.nn.BCELoss()
        self.measure = accuracy
        self.threshold = torch.tensor([threshold]).to(self.device)
        
        self.lin = Linear(num_sub_structs, 1)
                
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        return optimizer

    
    def training_step(self, batch, batch_idx):
        #Forward pass, #calculate loss and acc and loga
        num_graphs = batch["y"].shape[0]
        out = self.forward(batch["x"], num_graphs)
        loss, acc, unhealthy_acc = self.compute_metrics(out, batch["y"], unhealthy_seperate = True)
        
        #log
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        self.log('unhealthy_train_acc', unhealthy_acc)
        return loss

    
    def validation_step(self, batch, batch_idx):
        #Forward pass, #calculate loss and acc and loga
        num_graphs = batch["y"].shape[0]
        out = self.forward(batch["x"], num_graphs)
        loss, acc = self.compute_metrics(out, batch["y"])
        
        #log
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        return loss
    
    
    def test_step(self, batch, batch_idx):
        #Forward pass, #calculate loss and acc and loga
        num_graphs = batch["y"].shape[0]
        out = self.forward(batch["x"], num_graphs)
        loss, acc = self.compute_metrics(out, batch["y"])
        
        #log
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        return loss
    
    def compute_metrics(self, model_output, targets, unhealthy_seperate = False):
        # Make double and get loss
        model_output = model_output.squeeze().double()
        targets = targets.squeeze().double()
        loss = self.criterion(model_output, targets)
        
        # get threshold on Cuda and find accuracy
        self.threshold = self.threshold.to(self.device)
        binaries = (model_output>self.threshold).int()
        acc = self.measure(binaries, targets)
        if unhealthy_seperate:
            mask = (targets == 1)
            if mask.sum() > 0:
                unhealthy_acc = self.measure(binaries[mask], targets[mask])
            else:
                unhealthy_acc = 0
            return loss, acc, unhealthy_acc
        
        return loss, acc


    def forward(self, batch, num_graphs):
        embeddings = torch.empty((num_graphs,0)).to(self.device)
        for i, graph in enumerate(batch):
            x = self.all_gnns[i](graph)
            embeddings = torch.hstack((embeddings, x))
        
        x = embeddings.squeeze().reshape(-1, self.n_sub_structs)
        x = self.lin(x)
        x = x.sigmoid()
        
        return x


###########NEXT MODEL ################################################################################################################################################################################################################################################


class Oasis_PCA_Net(pl.LightningModule):
    def __init__(self, hidden_channels = 64, num_node_features = 3, 
                 num_classes = 2, conv_layer = GraphConv, num_sub_structs = 15, 
                 pca_components = 10):
        super(Oasis_PCA_Net, self).__init__()
        
        self.n_sub_structs = num_sub_structs
        self.num_classes = num_classes
        self.pca_components = pca_components
        
        self.criterion = torch.nn.CrossEntropyLoss()
        self.measure = accuracy
        
        
    def fit_pcas(self, dataset):
        #set up list of lists
        pts_list = []
        pca_list = []
        for i in range(self.n_sub_structs):
            new_list = []
            pts_list.append(new_list)
            if self.pca_components > 1:
                new_pca = decomp.PCA(n_components = self.pca_components)
            else:
                new_pca = decomp.PCA(n_components = self.pca_components, svd_solver = 'full')
            pca_list.append(new_pca)

        # get datpoints into lists

        print("Preparing PCA data and fitting...")
        for i, item in tqdm(enumerate(dataset), total = len(dataset)):
            data_ = item["x"]
            for i, substructure in enumerate(data_):
                v = substructure.pos
                v = np.hstack((v[:,0], v[:,1], v[:,2]))
                pts_list[i].append(v)

        for i, _ in enumerate(pts_list):
            pts_list[i] = self.pca_helper_data_transform(pts_list[i])

        # fit the PCAs
        for i, _ in enumerate(pca_list):
            data = pts_list[i]
            pca_list[i].fit(data)
        self.pca_list = pca_list
        print("done!")
        
        total_components = 0
        for pca in self.pca_list:
            total_components+= pca.n_components_
        
        self.lin2 = Linear(total_components, self.num_classes)
    
    def pca_helper_data_transform(self, pts):
        if isinstance(pts, list):
            pts = torch.tensor(pts)

        pts = pts.T
        # convert to list of np_arrays and set up for PCA
        m, n = pts.shape
        num_nodes = m//3;
        x_ind = range(num_nodes)
        y_ind = range(num_nodes,num_nodes*2)
        z_ind = range(num_nodes*2,num_nodes*3)
        cx = pts[x_ind,:];
        cy = pts[y_ind,:];
        cz = pts[z_ind,:];
        # spatial normalisation
        cx_norm = cx - torch.tile(torch.mean(cx,axis=0),(num_nodes,1))
        cy_norm = cy - torch.tile(torch.mean(cy,axis=0),(num_nodes,1))
        cz_norm = cz - torch.tile(torch.mean(cz,axis=0),(num_nodes,1))
        X = torch.vstack((cx_norm, cy_norm, cz_norm))
        X = X.T
        return X
        
    
    def forward(self, batch, num_graphs):
        batch_size = int(batch[0].y.shape[0])
        pts_list = []        
        
        for i, substructure in enumerate(batch):
                v = substructure.pos
                substruct_num_nodes = int(v.shape[0]/batch_size)
                tmp_list = torch.empty((0,substruct_num_nodes*3)).to(self.device)
                for i in range(batch_size):
                    start_index = int(i * substruct_num_nodes)
                    end_index = start_index + substruct_num_nodes
                    sample_nodes_flattened = torch.hstack((v[start_index:end_index,0], 
                                                           v[start_index:end_index,1], 
                                                           v[start_index:end_index,2]))
                    sample_nodes_flattened = sample_nodes_flattened.reshape(1, -1).to(self.device)
                    tmp_list = torch.vstack((tmp_list, sample_nodes_flattened))        
                pts_list.append(tmp_list)
        
        embeddings = torch.empty((batch_size, 0)).to(self.device)
        for i, points in enumerate(pts_list):
            X = self.pca_helper_data_transform(points)
            X = X.detach().cpu()
            x = self.pca_list[i].transform(X)
            x = torch.from_numpy(x).to(self.device)
            embeddings = torch.hstack((embeddings, x))
            
        x = self.lin2(embeddings)
        return x
    
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        return optimizer

    
    def training_step(self, batch, batch_idx):
        #Forward pass, #calculate loss and acc and loga
        num_graphs = batch["y"].shape[0]
        out = self.forward(batch["x"], num_graphs)
        loss, acc, unhealthy_acc = self.compute_metrics(out, batch["y"], unhealthy_seperate = True)
        
        #log
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        self.log('unhealthy_train_acc', unhealthy_acc)
        return loss

    
    def validation_step(self, batch, batch_idx):
        #Forward pass, #calculate loss and acc and loga
        num_graphs = batch["y"].shape[0]
        out = self.forward(batch["x"], num_graphs)
        loss, acc = self.compute_metrics(out, batch["y"])
        
        #log
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        return loss
    
    
    def test_step(self, batch, batch_idx):
        #Forward pass, #calculate loss and acc and loga
        num_graphs = batch["y"].shape[0]
        out = self.forward(batch["x"], num_graphs)
        loss, acc = self.compute_metrics(out, batch["y"])
        
        #log
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        return loss
    
    def compute_metrics(self, model_output, targets, unhealthy_seperate = False):
        loss = self.criterion(model_output, targets)
        acc = self.measure(model_output.argmax(dim = 1), targets)
        if unhealthy_seperate:
            mask = (targets == 1)
            if mask.sum() > 0:
                unhealthy_acc = self.measure(model_output[mask].argmax(dim = 1), targets[mask])
            else:
                unhealthy_acc = 0
            return loss, acc, unhealthy_acc
        
        return loss, acc




###########NEXT MODEL ################################################################################################################################################################################################################################################


class Oasis_PCA_Net_Binary(pl.LightningModule):
    def __init__(self, hidden_channels = 64, num_node_features = 3, conv_layer = GraphConv, num_sub_structs = 15, 
                 pca_components = 10, threshold = 0.5):
        super(Oasis_PCA_Net_Binary, self).__init__()
        
        self.n_sub_structs = num_sub_structs
        self.pca_components = pca_components
        
        self.criterion = torch.nn.BCELoss()
        self.measure = accuracy
        self.threshold = torch.tensor([threshold]).to(self.device)
        
        
    def fit_pcas(self, dataset):
        #set up list of lists
        pts_list = []
        pca_list = []
        for i in range(self.n_sub_structs):
            new_list = []
            pts_list.append(new_list)
            if self.pca_components > 1:
                new_pca = decomp.PCA(n_components = self.pca_components)
            else:
                new_pca = decomp.PCA(n_components = self.pca_components, svd_solver = 'full')
            pca_list.append(new_pca)

        # get datpoints into lists

        print("Preparing PCA data and fitting...")
        for i, item in tqdm(enumerate(dataset), total = len(dataset)):
            data_ = item["x"]
            for i, substructure in enumerate(data_):
                v = substructure.pos
                v = np.hstack((v[:,0], v[:,1], v[:,2]))
                pts_list[i].append(v)

        for i, _ in enumerate(pts_list):
            pts_list[i] = self.pca_helper_data_transform(pts_list[i])

        # fit the PCAs
        for i, _ in enumerate(pca_list):
            data = pts_list[i]
            pca_list[i].fit(data)
        self.pca_list = pca_list
        print("done!")
        
        total_components = 0
        for pca in self.pca_list:
            total_components+= pca.n_components_

        self.lin = Linear(total_components, 1)
        
    
    def pca_helper_data_transform(self, pts):
        if isinstance(pts, list):
            pts = torch.tensor(pts)

        pts = pts.T
        # convert to list of np_arrays and set up for PCA
        m, n = pts.shape
        num_nodes = m//3;
        x_ind = range(num_nodes)
        y_ind = range(num_nodes,num_nodes*2)
        z_ind = range(num_nodes*2,num_nodes*3)
        cx = pts[x_ind,:];
        cy = pts[y_ind,:];
        cz = pts[z_ind,:];
        # spatial normalisation
        cx_norm = cx - torch.tile(torch.mean(cx,axis=0),(num_nodes,1))
        cy_norm = cy - torch.tile(torch.mean(cy,axis=0),(num_nodes,1))
        cz_norm = cz - torch.tile(torch.mean(cz,axis=0),(num_nodes,1))
        X = torch.vstack((cx_norm, cy_norm, cz_norm))
        X = X.T
        return X
        
    
    def forward(self, batch, num_graphs):
        batch_size = int(batch[0].y.shape[0])
        pts_list = []        
        
        for i, substructure in enumerate(batch):
                v = substructure.pos
                substruct_num_nodes = int(v.shape[0]/batch_size)
                tmp_list = torch.empty((0,substruct_num_nodes*3)).to(self.device)
                for i in range(batch_size):
                    start_index = int(i * substruct_num_nodes)
                    end_index = start_index + substruct_num_nodes
                    sample_nodes_flattened = torch.hstack((v[start_index:end_index,0], 
                                                           v[start_index:end_index,1], 
                                                           v[start_index:end_index,2]))
                    sample_nodes_flattened = sample_nodes_flattened.reshape(1, -1).to(self.device)
                    tmp_list = torch.vstack((tmp_list, sample_nodes_flattened))        
                pts_list.append(tmp_list)
        
        embeddings = torch.empty((batch_size, 0)).to(self.device)
        for i, points in enumerate(pts_list):
            X = self.pca_helper_data_transform(points)
            X = X.detach().cpu()
            x = self.pca_list[i].transform(X)
            x = torch.from_numpy(x).to(self.device)
            embeddings = torch.hstack((embeddings, x))
            
        x = self.lin(embeddings)
        x = x.sigmoid()
        return x
    
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        return optimizer

    
    def training_step(self, batch, batch_idx):
        #Forward pass, #calculate loss and acc and loga
        num_graphs = batch["y"].shape[0]
        out = self.forward(batch["x"], num_graphs)
        loss, acc, unhealthy_acc = self.compute_metrics(out, batch["y"], unhealthy_seperate = True)
        
        #log
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        self.log('unhealthy_train_acc', unhealthy_acc)
        return loss

    
    def validation_step(self, batch, batch_idx):
        #Forward pass, #calculate loss and acc and loga
        num_graphs = batch["y"].shape[0]
        out = self.forward(batch["x"], num_graphs)
        loss, acc = self.compute_metrics(out, batch["y"])
        
        #log
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        return loss
    
    
    def test_step(self, batch, batch_idx):
        #Forward pass, #calculate loss and acc and loga
        num_graphs = batch["y"].shape[0]
        out = self.forward(batch["x"], num_graphs)
        loss, acc = self.compute_metrics(out, batch["y"])
        
        #log
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        return loss
    
    def compute_metrics(self, model_output, targets, unhealthy_seperate = False):
        # Make double and get loss
        model_output = model_output.squeeze().double()
        targets = targets.squeeze().double()
        loss = self.criterion(model_output, targets)
        
        # get threshold on Cuda and find accuracy
        self.threshold = self.threshold.to(self.device)
        binaries = (model_output>self.threshold).int()
        acc = self.measure(binaries, targets)
        if unhealthy_seperate:
            mask = (targets == 1)
            if mask.sum() > 0:
                unhealthy_acc = self.measure(binaries[mask], targets[mask])
            else:
                unhealthy_acc = 0
            return loss, acc, unhealthy_acc
        
        return loss, acc





"""
MLP class in case needed for embedding into larger model
"""
class MiniMlp_Binary(torch.nn.Module):
    def __init__(self, input_size):
        super(MiniMlp_Binary, self).__init__()
        mid_layer = int(max(2, input_size/2))
        self.mod = Sequential(
            Linear(input_size, mid_layer),
            LinearDropout(0.5),
            ReLU(),
            Linear(mid_layer, 1),
            Sigmoid()
          )
    
    def forward(self, x):
        return self.mod(x)
    












