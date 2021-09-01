from tqdm import tqdm
import vtk
import numpy as np
import pyvista as pv
from pathlib import Path
from  torch_geometric.data import Dataset, Data
import pickle

import os
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from torch_geometric.data import DataLoader as PyGDataLoader

all_sub_structs_list = ['L_Pall', 'R_Pall', 'L_Caud', 'R_Caud', 'R_Hipp', 'L_Hipp', 'R_Amyg', 
                   'L_Amyg', 'R_Thal', 'BrStem', 'L_Thal', 'L_Puta', 'R_Puta', 'R_Accu', 
                   'L_Accu']


            
###########NEXT MODEL################################################################################################################################################################################################################################################

class Oasis_Dataset_static(Dataset):
    """
    PyTorch Geometric Dataset for handling OASIS data in static form 
    (i.e. CDR and image matched without looking at patients future development
    """
    def __init__(self, 
                 root: str = "/vol/biomedic3/bglocker/brain/oasis3/rigid_to_mni/seg/fsl/meshes/", 
                 labels_feats_path: str = "multi_per_sub_single_per_scan_file",
                 substructures: List[str] = all_sub_structs_list, 
                 transform=None, pre_transform=None, 
                 cache_path: str = './cached_files', reload_path: bool = False,
                 cache_file_path = "static_oasis_multi_per_sub_cache",
                 max_normal_threshold:int = 0
                ):
        super(Oasis_Dataset_static, self).__init__(root, transform, pre_transform)
        self.root = root
        self.transform = transform
        
        self.max_normal_threshold = max_normal_threshold
        
        # Setup labels pandas
        labels_feats_path = os.path.join(cache_path, "statics", labels_feats_path)
        with open(labels_feats_path, 'rb') as flat_list_file:
                self.features_df_container = pickle.load(flat_list_file)
                
        self.substructures = sorted(substructures)
        self.flat_list = []
        self.reload_path = reload_path
        self.cache_path = cache_path
        self.flat_list_file = os.path.join(self.cache_path, cache_file_path)
        self.__read_path_structure()
        
    
    def __read_path_structure(self):
        if not self.reload_path:
            with open(self.flat_list_file, 'rb') as flat_list_file:
                self.flat_list = pickle.load(flat_list_file)
            print(f'\n...loaded flat_list from {self.flat_list_file}', end = " ")
            print(f' with {len(self.flat_list)} subjects...')
            return
        
        self.data_subject_ids = self.get_all_subject_ids()
        
        for _id, _day in tqdm(self.data_subject_ids):
            tmp_list = []
            for substructure in self.substructures:
                full_path = self.create_vtk_path(_id, _day, substructure, False)
                if not os.path.exists(full_path):
                    full_path = self.create_vtk_path(_id, _day, substructure, True)
                    if not os.path.exists(full_path):
                        continue
                tmp_list.append(full_path)
            if tmp_list:
                self.flat_list.append(tmp_list)

        with open(self.flat_list_file, 'wb') as flat_list_file:
            pickle.dump(self.flat_list, flat_list_file)
        
        print(f'\n...freshly loaded flat_list from {self.flat_list_file}', end = " ")
        print(f'with {len(self.flat_list)} subjects...')
        return
    
    def create_vtk_path(self, _id, _day, substructure, run_01):
        if not run_01:
            patient_substruct =  f"sub-{_id}_ses-{_day}_T1w_unbiased-{substructure}_first.vtk"
        else:
            patient_substruct =  f"sub-{_id}_ses-{_day}_run-01_T1w_unbiased-{substructure}_first.vtk"
        return os.path.join(self.root, patient_substruct)
    
    def get_all_subject_ids(self):
        all_ids = list(self.features_df_container["MR ID_MR"])
        ids = [[a[:8], a[-5:]] for a in all_ids]
        return ids
    
    
    def get(self, idx: int):
        data_subject_id, meshes, targets, original_target = self.get_raw(idx)
        targets = torch.tensor(targets) 
        data_dict = {"polydata": meshes, "targets": targets, "id": data_subject_id}
        return data_dict
    
    def get_id_from_vtk_path(self, vtk_path):
        id_start = vtk_path.find('OAS')
        id_number = vtk_path[id_start:id_start+8]
        day_start = vtk_path.find('ses-d')
        day_number = vtk_path[day_start+4:day_start+9]
        return id_number, day_number
    
    def get_pv_mesh_from_path(self, full_path: str): 
        return pv.read(full_path) 
    
    def get_raw(self, idx: int):
        #get path at index and related id
        paths = self.flat_list[idx]
        data_subject_id, day_id = self.get_id_from_vtk_path(paths[0])

        #get mesh and features for that id
        meshes = []
        for path in paths:
            meshes.append(self.get_pv_mesh_from_path(path))
        targets, original_target = self.get_patient_id_targets(data_subject_id, day_id)
        return data_subject_id, meshes, targets, original_target
    
    def get_patient_id_targets(self, data_subject_id, day_id):
        MR_id = data_subject_id+"_MR_"+day_id
        mask = (self.features_df_container["MR ID_MR"] == MR_id)
        targets = self.features_df_container.loc[mask]
        target = int(targets["cdr"])
        bin_target = 1 if target > self.max_normal_threshold else 0
        return bin_target, target
    
    def __len__(self):
        return len(self.flat_list)




            
###########NEXT MODEL################################################################################################################################################################################################################################################




class Oasis_Dataset(Dataset):
    """
    PyTorch Geometric Dataset for handling OASIS data
    """
    def __init__(self, 
                 root: str = "/vol/biomedic3/bglocker/brain/oasis3/rigid_to_mni/seg/fsl/meshes/", 
                 labels_feats_path: str = "oasis_test_subjects_file",
                 substructures: List[str] = all_sub_structs_list, 
                 transform=None, pre_transform=None, 
                 cache_path: str = './cached_files', reload_path: bool = False,
                 cache_file_path = "all_subs_flat_list_file"
                ):
        super(Oasis_Dataset, self).__init__(root, transform, pre_transform)
        self.root = root
        self.transform = transform
        
        # Setup labels pandas
        labels_feats_path = os.path.join(cache_path, labels_feats_path)
        with open(labels_feats_path, 'rb') as flat_list_file:
                self.features_df_container = pickle.load(flat_list_file)
                
        self.substructures = sorted(substructures)
        self.flat_list = []
        self.reload_path = reload_path
        self.cache_path = cache_path
        self.flat_list_file = os.path.join(self.cache_path, cache_file_path)
        self.__read_path_structure()
        
    
    def __read_path_structure(self):
        if not self.reload_path:
            with open(self.flat_list_file, 'rb') as flat_list_file:
                self.flat_list = pickle.load(flat_list_file)
            print(f'\n...loaded flat_list from {self.flat_list_file}', end = " ")
            print(f' with {len(self.flat_list)} subjects...')
            return
        
        self.data_subject_ids = self.get_all_subject_ids()

        for _id, _day in tqdm(self.data_subject_ids):
            tmp_list = []
            for substructure in self.substructures:
                full_path = self.create_vtk_path(_id, _day, substructure, False)
                if not os.path.exists(full_path):
                    full_path = self.create_vtk_path(_id, _day, substructure, True)
                    if not os.path.exists(full_path):
                        continue
                tmp_list.append(full_path)
            if tmp_list:
                self.flat_list.append(tmp_list)
        
        #cache file for faster loading next time        
        with open(self.flat_list_file, 'wb') as flat_list_file:
            pickle.dump(self.flat_list, flat_list_file)
        
        print(f'\n...freshly loaded flat_list from {self.flat_list_file}', end = " ")
        print(f'with {len(self.flat_list)} subjects...')
        return
    
    def create_vtk_path(self, _id, _day, substructure, run_01):
        if not run_01:
            patient_substruct =  f"sub-{_id}_ses-{_day}_T1w_unbiased-{substructure}_first.vtk"
        else:
            patient_substruct =  f"sub-{_id}_ses-{_day}_run-01_T1w_unbiased-{substructure}_first.vtk"
        return os.path.join(self.root, patient_substruct)
    
    def get_all_subject_ids(self):
        all_ids = list(self.features_df_container["First_MR_ID"])
        ids = [[a[:8], a[-5:]] for a in all_ids]
        return ids
    

    def get(self, idx: int):
        data_subject_id, meshes, targets = self.get_raw(idx)
        targets = torch.tensor(targets.to_numpy()) 
        data_dict = {"polydata": meshes, "targets": targets, "id": data_subject_id}
        return data_dict
    
    def get_id_from_vtk_path(self, vtk_path):
        id_start = vtk_path.find('OAS')
        return vtk_path[id_start:id_start+8]
    
    def get_pv_mesh_from_path(self, full_path: str): 
        return pv.read(full_path) 
    
    def get_raw(self, idx: int):
        #get path at index and related id
        paths = self.flat_list[idx]
        data_subject_id = self.get_id_from_vtk_path(paths[0])

        #get mesh and features for that id
        meshes = []
        for path in paths:
            meshes.append(self.get_pv_mesh_from_path(path))
        targets = self.get_patient_id_targets(data_subject_id)
        return data_subject_id, meshes, targets  
    
    def get_patient_id_targets(self, data_subject_id):
        mask = (self.features_df_container["Subject"] == data_subject_id)
        targets = self.features_df_container.loc[mask]
        return targets["Label"]
    
    def __len__(self):
        return len(self.flat_list)
    







    