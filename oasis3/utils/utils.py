import numpy as np
import vtk


"""Data dictionaries"""

all_sub_structs_list = ['L_Pall', 'R_Pall', 'L_Caud', 'R_Caud', 'R_Hipp', 'L_Hipp', 'R_Amyg', 
                   'L_Amyg', 'R_Thal', 'BrStem', 'L_Thal', 'L_Puta', 'R_Puta', 'R_Accu', 
                   'L_Accu']

translate = { "BrStem" : "Brainstem", "Hipp" : "Hippocampus",  "Amyg" : "Amygdala", 
             "Pall" : "Pallidum", "Caud" : "Caudate", "Puta" : "Putamen", 
             "Accu" : "Accumbens", "Thal" : "Thalamus"} 



"""Umeyama transformation helpers"""

def umeyama_similarity(X, Y):
    
    # Get dimension and number of points
    m, n = X.shape

    # Demean the point sets X and Y
    X_mean = X.mean(1) #MODEL ANSWER
    Y_mean = Y.mean(1) #MODEL ANSWER
    
    X_demean =  X - np.tile(X_mean, (n, 1)).T #MODEL ANSWER
    Y_demean =  Y - np.tile(Y_mean, (n, 1)).T #MODEL ANSWER

    # Computing matrix XY' using demeaned and NORMALISED point sets (divide by the number of points n)
    # See Equation (38) in the paper
    XY = np.dot(X_demean, Y_demean.T) / n  #MODEL ANSWER

    # Determine variances of points X and Y, see Equation (36),(37) in the paper    
    X_var = np.mean(np.sum(X_demean*X_demean, 0))
    Y_var = np.mean(np.sum(Y_demean*Y_demean, 0))

    # Singular value decomposition
    U,D,V = np.linalg.svd(XY,full_matrices=True,compute_uv=True)
    V=V.T.copy()
    
    # Determine rotation
    R = np.dot( V, U.T) #MODEL ANSWER
    
    # Determine the scaling, see Equation (42) in the paper (assume S to be the identity matrix, so ignore)
    c = np.trace(np.diag(D)) / X_var #MODEL ANSWER

    # Determine translation, see Equation (41) in the paper
    t = Y_mean - c * np.dot(R, X_mean) #MODEL ANSWER

    return R,t,c


def umeyama_rigid(X, Y):
    
    # Get dimension and number of points
    m, n = X.shape
    
    # Demean the point sets X and Y
    X_mean = X.mean(1)
    Y_mean = Y.mean(1)

    X_demean =  X - np.tile(X_mean, (n, 1)).T
    Y_demean =  Y - np.tile(Y_mean, (n, 1)).T
    
    # Computing matrix XY' using demeaned point sets
    XY = np.dot(X_demean, Y_demean.T)

    # Singular value decomposition
    U,D,V = np.linalg.svd(XY,full_matrices=True,compute_uv=True)
    V=V.T.copy()
    
    # Determine rotation
    R = np.dot( V, U.T)

    # Determine translation
    t = Y_mean - np.dot(R, X_mean)
    
    return R,t
    
    
def umeyama_transform_data(X, Y, mode, num_nodes):
    if mode == 'rigid':
            R, t = umeyama_rigid(X, Y)        
            warped = np.dot(R,X) + np.tile(t, (num_nodes, 1)).transpose()       
    elif mode == 'similar':
        R, t, c = umeyama_similarity(X, Y)
        warped = c * np.dot(R,X) + np.tile(t, (num_nodes, 1)).transpose()
        
    return warped


def demean_points(points):
    avg = points.mean(axis = 0)
    return points - avg


""" Read VTK file helper """

def read_vtk(filename):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()
    polydata = reader.GetOutput()
    vertices = np.array([polydata.GetPoint(i) for i in range(polydata.GetNumberOfPoints())])
    return vertices