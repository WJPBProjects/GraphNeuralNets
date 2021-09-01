from utils.utils import umeyama_transform_data, demean_points
from utils.ukbb_transforms import UKBB_all_structs_default, all_structs_default_plus_umeyama


import torch_geometric.transforms as PyG_T
import pytorch_lightning as pl
from typing import List, Optional
import pickle

import vtk
import numpy as np
import pyvista as pv
from pathlib import Path
from torch.utils.data.dataset import Dataset
from torch_geometric.transforms import Compose
import os
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from torch_geometric.data import Data
from torch_geometric.data import DataLoader as PyGDataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool


from torch.utils.data import WeightedRandomSampler

from Oasis_Dataset_file import Oasis_Dataset, all_sub_structs_list, Oasis_Dataset_static

class Oasis_PL(pl.LightningDataModule):
    """
    Lightning module to hold OASIS data
    """
    def __init__(self, 
                 data_dir:str = "/vol/biomedic3/bglocker/brain/oasis3/rigid_to_mni/seg/fsl/meshes/", 
                 labels_feats_path: str = "oasis_test_subjects_file",
                 substructures: List[str] = all_sub_structs_list, 
                 transform = UKBB_all_structs_default(), 
                 pre_transform=None, 
                 cache_path: str = './cached_files',
                 cache_file_path = "all_subs_flat_list_file",
                 reload_path: bool = False, 
                 batch_size: int = 4, 
                 train_test_val_split: List[int] = [70, 20,10], 
                 num_workers = 8,
                 balance_classes = False,
                 static = False
                ):
        super().__init__()
            
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transform
        self.pre_transform = pre_transform
        self.ttv_split = train_test_val_split
        self.num_workers = num_workers
        self.reload_path = reload_path
        self.cache_file_path = cache_file_path
        
        self.cache_path = cache_path
        self.labels_feats_path = labels_feats_path
        self.substructures = sorted(substructures)
        
        self.balance_classes = balance_classes
        self.static = static
        
        
        
    #Initialises the dataset
    def prepare_data(self):
        print("Initialising underlying dataset...", end = " ")
        if not self.static:
            self._data = Oasis_Dataset(
                root = self.data_dir, 
                labels_feats_path = self.labels_feats_path,
                substructures = self.substructures,
                transform = self.transform, 
                pre_transform = self.pre_transform, 
                cache_path = self.cache_path, 
                reload_path = self.reload_path,
                cache_file_path = self.cache_file_path
            )
        else:
            self._data = Oasis_Dataset_static(
                root = self.data_dir, 
                substructures = self.substructures,
                transform = self.transform, 
                pre_transform = self.pre_transform, 
                cache_path = self.cache_path, 
                reload_path = self.reload_path,
            )
        print("done! \n")
        

    # Sets up the split of data in test train and val
    def setup(self, stage: Optional[str] = None, reload_sampler = True):
        
        # Split into train_test_val
        n = len(self._data)
        ttv = np.array(self.ttv_split)/100
        n_train, n_test, _  = (n*ttv).astype(int)
        n_val = n - (n_train + n_test) # done so all graphs are used
        
        #Initialise random split of data
        self.train_set, self.test_set, self.val_set = \
        torch.utils.data.random_split(self._data,[n_train, n_test, n_val])
        
        # If samplers need reloading
        if reload_sampler and stage in ['fit', None]:
            if self.balance_classes:
                print("...rebalancing dataset...")
                # Get class counts
                print("train...")
                self.train_uniques, self.train_labels = self.class_sample_count(self.train_set)
                print("done! Now val...")
                self.val_uniques, self.val_labels = self.class_sample_count(self.val_set)

                # Set up weighted samplers
                self.train_sampler = self.get_sampler_equalised(self.train_uniques, self.train_labels)
                self.val_sampler = self.get_sampler_equalised(self.val_uniques, self.val_labels)
                print("done!")
        if not reload_sampler:
            with open("./cached_files/statics/train_sampler_file", 'rb') as sampler:
                samplers = pickle.load(sampler)
                self.train_sampler = samplers[0]
                self.train_sampler = samplers[1]
    
    def set_umeyama(self, stage1 = 'demean', stage2 = 'rigid', n_examples = 100):
        print("Calculating reference shape(s) for umeyama")
        avgs = self.get_avg_shape_demeaned_multi_structs(n_examples)
        t = all_structs_default_plus_umeyama(avgs, stage2)
        self._data.transform = self.transform = t
    
    def train_dataloader(self):
        if self.balance_classes:
            return PyGDataLoader(self.train_set, batch_size=self.batch_size,
                                 num_workers = self.num_workers, sampler = self.train_sampler)
        return PyGDataLoader(self.train_set, batch_size=self.batch_size, 
                             shuffle=True, num_workers = self.num_workers)

    def val_dataloader(self):
        if self.balance_classes:
            return PyGDataLoader(self.train_set, batch_size=self.batch_size,
                                 num_workers = self.num_workers, sampler = self.val_sampler)
        return PyGDataLoader(self.val_set, batch_size=self.batch_size,
                             num_workers = self.num_workers)

    def test_dataloader(self):
        return PyGDataLoader(self.test_set, batch_size=self.batch_size, 
                             num_workers = self.num_workers)

    def class_sample_count(self, X_set):
        labels = torch.empty((0,1))
        for item in tqdm(X_set):
            targ = item["y"]
            labels = torch.vstack((labels, targ))
        labels = labels.detach().cpu().numpy().astype(int)
        return np.unique(labels, return_counts=True), labels
    
    # set up weighted sampler
    def get_sampler_equalised(self, np_unique_w_counts, target, altered_class = 1):
        unaltered_class = 0 if altered_class == 1 else 1    
        
        # match class sample sizes
        class_counts = np_unique_w_counts[1]
        
        # Calc weights
        weight = 1. / class_counts
        samples_weight = weight[target]
        samples_weight = samples_weight.squeeze()
        samples_weight = torch.from_numpy(samples_weight)
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        return sampler
    
    
    def get_avg_shape_demeaned_multi_structs(self, n_examples):
        avgs = []
        
        for i in tqdm(range(n_examples)):
            substructs = self._data[i]["x"]
            for j, substruct in enumerate(substructs):
                points = substruct.pos
                points = demean_points(points)
                if i ==0:
                    avgs.append(points)
                else:
                    avgs[j] = np.dstack([avgs[j], points])
        
        for i, _ in enumerate(avgs):
            avgs[i] = np.mean(avgs[i], axis = 2)
        return avgs
                    
            