import torch
import pytorch_lightning as pl

from torch_geometric.nn import GCNConv, GraphConv, global_mean_pool, avg_pool_neighbor_x

import torch.nn.functional as F
from torch.nn import Linear

from torch_geometric.utils import accuracy




class MNIST_Graph_Net(pl.LightningModule):
    def __init__(self, hidden_channels = 64, num_node_features = 3, num_classes = 2, conv_layer = GraphConv, CBS_initial_neighb_distance = 0, CBS_epochs = 1, starting_own_weight = 0.5):
        super().__init__()
        
        self.conv1 = conv_layer(num_node_features, hidden_channels)
        self.conv2 = conv_layer(hidden_channels, hidden_channels)
        self.conv3 = conv_layer(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)
        
        self.avg_pool = avg_pool_neighbor_x
        
        self.criterion = torch.nn.CrossEntropyLoss()
        self.measure = accuracy
        
        self.CBS = CBS_initial_neighb_distance
        self.CBS_epochs = CBS_epochs
        
        self.own_weight = starting_own_weight
        if self.CBS:
            self.weight_incr = (1 - starting_own_weight)/self.CBS
       
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        return optimizer
    
    
    def forward(self, d):
        # 1. Obtain node embeddings 
        d.x = self.conv1(d.x, d.edge_index)
        if self.training:
            self.smooth_data_x2(d)
        d.x = d.x.relu()
        
        d.x = self.conv2(d.x, d.edge_index)
        if self.training:
            self.smooth_data_x2(d)
        d.x = d.x.relu()
        
        d.x = self.conv3(d.x, d.edge_index)

        # 2. Readout layer
        x = global_mean_pool(d.x, d.batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x
    
    
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
        loss = self.criterion(model_output, targets)
        acc = self.measure(model_output.argmax(dim=1), targets)
        return loss, acc
    
    def on_train_epoch_end(self):    
        if self.CBS:
            if ((self.current_epoch+1) % self.CBS_epochs) == 0:
                self.CBS -= 1
                print(f"now using CBS loop of {self.CBS}")
                
                self.own_weight += self.weight_incr
                if self.CBS == 0:
                    self.own_weight = 1
                print(f"now using self weight of {self.own_weight}")
                
    def smooth_data_x(self, data):
        for i in range(self.CBS):
            self.avg_pool(data)
            
    def weighted_avg_data(self, data):
        x_tmp = data.x
        self.avg_pool(data)
        data.x = (self.own_weight * x_tmp) + ((1-self.own_weight)*data.x)

    def smooth_data_x2(self, data):
        x_tmp = data.x
        for i in range(self.CBS):
            self.avg_pool(data)
        data.x = (self.own_weight * x_tmp) + ((1-self.own_weight)*data.x)