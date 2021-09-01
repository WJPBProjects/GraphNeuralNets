"""
File of data transforms to apply to OASIS data, based on UKBB transforms
"""

from .utils import umeyama_transform_data

import torch
from typing import Dict
import numpy as np
from torch_geometric.data import Data
from torch_geometric.transforms import Compose, ToUndirected,\
FaceToEdge, Center, NormalizeRotation,\
Spherical, Cartesian, LocalCartesian, Polar, Constant, AddSelfLoops

    
"""
Default/prebuilt Transforms
"""
class ukbb_to_undirected_PyG_edges:
    def __call__(self, data: Dict):
        self.transform = Compose([
            poly_to_PyG_faces(),
            FaceToEdge(remove_faces=False),
            ToUndirected()
        ])
        return self.transform(data)

class basic_add_y_as_feat:
    def __call__(self, data: Dict):
        self.transform = Compose([
            ukbb_to_undirected_PyG_edges(),
            y_as_node_feature()
        ])
        return self.transform(data)


class default_ukbb_transform:
    def __call__(self, data: Dict):
        self.transform = Compose([
            ukbb_to_undirected_PyG_edges(),
            pos_as_node_feature_x(),
        ])
        return self.transform(data)
    
class basic_add_normals_as_feat:
    def __call__(self, data: Dict):
        self.transform = Compose([
            ukbb_to_undirected_PyG_edges(),
            normals_as_node_feature_x(),
        ])
        return self.transform(data)

class default_plus_umeyama_transform:
    def __init__(self, avg_shape, mode: str = 'similar'):
        self.avg_shape = avg_shape
        if mode not in ['rigid', 'similar']:
            raise ValueError('Umeyama must be <rigid> or <similar>')
        self.mode = mode
        
    
    def __call__(self, data: Dict):
        self.transform = Compose([
            ukbb_to_undirected_PyG_edges(),
            umeyama_transform(self.avg_shape, self.mode),
            pos_as_node_feature_x(),
        ])
        return self.transform(data)

class coordinate_edge_attr:
    def __init__(self, mode: str, only_dummy_node_feat = True, add_normals = False):
        if mode == 'Spherical':
            self.coord = Spherical
        elif mode == 'Cartesian':
            self.coord = Cartesian
        elif mode == 'LocalCartesian':
            self.coord = LocalCartesian
        elif mode == 'Polar':
            self.coord = Polar
        else:
            print("Mode not valid. Setting edge_attr as Cartesian")
            self.coord = Cartesian
        
        if only_dummy_node_feat:
            self.add_feats = Constant()
        else:
            self.add_feats = pos_as_node_feature_x()
            
        if add_normals:
            self.add_normals = normals_as_node_feature_x()
        else:
            self.add_normals = None
            
    
    def __call__(self, data: Dict):
        self.transform = Compose([
           ukbb_to_undirected_PyG_edges(),
           self.coord(),
           self.add_feats,
        ])
        data = self.transform(data)
        data.x = data.x.float()
        data.edge_attr = data.edge_attr.float()
        return data
    
    
class UKBB_all_structs_default:
    def __init__(self):
        self.trans = get_PyG_data_from_polydata_and_features_all_substructs()
    def __call__(self, data):
        poly = data["polydata"]
        targets = data["targets"]
        data_final = {}
        data_final["y"] = targets.squeeze()
        data_final["x"] = self.trans(poly, targets)
        return data_final
    
    
class all_structs_default_plus_umeyama:
    def __init__(self, avg_shapes, mode: str = 'rigid'):
        if mode not in ['rigid', 'similar']:
            raise ValueError('Umeyama must be <rigid> or <similar>')

        self.umeyama_transform_list = []
        for shape in avg_shapes:
            new_umeyama = umeyama_transform(shape, mode)
            self.umeyama_transform_list.append(new_umeyama)
                                     
        self.trans = complete_multi_struct_transform(self.umeyama_transform_list)
        
    def __call__(self, data):
        poly = data["polydata"]
        targets = data["targets"]
        data_final = {}
        data_final["y"] = targets.squeeze()
        data_final["x"] = self.trans(poly, targets)
        return data_final



class complete_multi_struct_transform:
    def __init__(self, umeyama_list):
        self.prep = get_PyG_data_from_polydata_and_features()
        self.umeyamas = umeyama_list
        
        self.transform = Compose([
            FaceToEdge(remove_faces=False),
            ToUndirected(),
            pos_as_node_feature_x(),
            Spherical()
        ])
        
    
    def __call__(self, poly, targets):
        poly_list = []
        for i, polydata in enumerate(poly):
            data = self.prep(polydata, targets)
            data = self.umeyamas[i](data)
            data = self.transform(data)
            poly_list.append(data)
        return poly_list 
    
    
"""
Helper transforms start here
"""

#transform that returns Data object with faces (but not edges)
class poly_to_PyG_faces:
    def __call__(self, data: Dict):
        poly = data["polydata"]
        targets = data["targets"]
        path = data["path"]
        data = get_PyG_data_from_polydata_and_features(poly, targets, path)
        return data


# makes the node positions features of the nodes
class pos_as_node_feature_x:
    def __call__(self, data):
        to_add = data.pos.detach().clone()
        if data.x is None:
            data.x = to_add.double()
        else:
            data.x = torch.cat((data.x, to_add), dim = 1).double()
        return data
    
class normals_as_node_feature_x:
    def __call__(self, data):
        to_add = data.normals.detach().clone()
        if data.x is None:
            data.x = to_add.double()
        else:
            data.x = torch.cat((data.x, to_add), dim = 1).double()
        return data

# Umeyama Transform; initialised with an average shape to fit to; alters Data pos to fit shape
class umeyama_transform:
    def __init__(self, avg_shape, mode: str = 'similar'):
        self.avg_shape = avg_shape
        if mode not in ['rigid', 'similar']:
            raise ValueError('Umeyama must be <rigid> or <similar>')
        self.mode = mode

    def __call__(self, data: Data):        
        X = np.array(data.pos.T)
        assert X.shape == self.avg_shape.T.shape
        num_nodes = X.shape[1]
        warped = umeyama_transform_data(X, self.avg_shape.T, self.mode, num_nodes)
        data.pos = torch.tensor(warped.T)
        return data
        
        
class y_as_node_feature():
    def __call__(self, data):
        y = data.y.double()
        self.transform = Constant(y)
        self.transform(data)
        data.x = data.x.double()
        return data
        
        
        

"""
Helper functions
"""
class get_PyG_data_from_polydata_and_features:
    def __call__(self, polydata, targets, path = None):
        #Extract trianglular faces
        assert polydata.is_all_triangles()
        faces = polydata.faces.reshape(-1, 4)[:, 1:].T
        normals = polydata.point_normals
        pos = polydata.points

        data = Data(
            pos=torch.tensor(pos).double(),
            normals=torch.tensor(normals).double(),
            face=torch.tensor(faces),
            y = targets.squeeze().double(),
            path = path
        )
        return data

    
class get_PyG_data_from_polydata_and_features_all_substructs:
    def __init__(self):
        self.prep = get_PyG_data_from_polydata_and_features()
        self.transform = Compose([
            FaceToEdge(remove_faces=False),
            ToUndirected(),
            pos_as_node_feature_x()
        ])
        
    
    def __call__(self, poly, targets):
        poly_list = []
        for polydata in poly:
            data = self.prep(polydata, targets)
            data = self.transform(data)
            poly_list.append(data)
        return poly_list

    
    
    
    
    
    
    
    
    
    
    
    
    