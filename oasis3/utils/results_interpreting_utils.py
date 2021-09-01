import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pickle

from Oasis_DataModule_file import Oasis_PL
from UKBB_All_Substructs_Net_file import UKBB_All_Substructs_Net2, Oasis_PCA_Net, UKBB_All_Substructs_Net2_Binary

from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, roc_auc_score, auc, precision_recall_fscore_support, matthews_corrcoef

import os
import torch
import matplotlib.pyplot as plt

import pickle
import numpy as np


"""
Helpers
"""
# Get average confusion matrix from list
def get_avg_conf_matrix(list_of_matrixes):
    matrix = list_of_matrixes[0]
    for matrix_ in list_of_matrixes[1:]:
        matrix += matrix_
    matrix = matrix / matrix.sum()
    matrix = np.around(matrix, 3)*100
    return matrix

# Get average and std of accuracies from list
def get_avg_std_acc(list_of_accs):
    num = len(list_of_accs)
    total = 0
    for acc in list_of_accs:
        total += acc
    avg = total/num
    return avg, std


# Print stastics to screen
def do_printing(matrices, accs, balanced_accs, precs, recs, f1s, mccs, specs, aucs):
    normalised_matrix = get_avg_conf_matrix(matrices)
    print("Normalised average matrix \n", normalised_matrix, "\n")
    
    scores = [accs, balanced_accs, precs, recs, f1s, mccs, specs, aucs]
    text = ["Accuracy: ", "Balanced acc. : ", "Precision: ", "Recall: ", "F1: ", "Specificity: ", \
            "MCC: ", "AUC: "]
    
    for name, item in zip(text, scores):
        item = np.array(item)
        print(name, "{0:.4f}".format(item.mean()), "±", "{0:.4f}".format(item.std()), "\n")

#Returns list of empty lists
def get_empty_list(num = 8):
    x = []
    for i in range(num):
        x.append([])
    return x

# Finds minimum distance between avg_tpr and avg_fpr and their respetive lists fprs and tprs
def min_distance(fprs, tprs, avg_fpr, avg_tpr):
    dist_fpr = (fprs-avg_fpr)**2
    dist_tpr = (tprs - avg_tpr)**2
    dist_total = dist_fpr + dist_tpr
    min_i = np.argmin(dist_total)
    return fprs[min_i], tprs[min_i]


"""
Functiont to print stuff or draw curves
"""
# Draws ROC curve given by fprs and tprs on the axis (ax) given to it
def draw_ROC_curves(fprs, tprs, auc_scores_list, ax, type_name, recs, specs):
    # get point for current threshold
    point5_tprs = np.array(recs)
    point5_fprs = 1 - np.array(specs)
    avg_tpr = point5_tprs.mean()
    avg_fpr = point5_fprs.mean()
    
    if ax is None:
        ax = plt.gca()
    lw_main = 3
    lw_sub = 0.5
    
    # get means
    fprs_mean = fprs.mean(axis = 0)
    tprs_mean = tprs.mean(axis = 0)
    auc_mean = auc(fprs_mean, tprs_mean)
    ax.plot(fprs_mean, tprs_mean, color='darkorange', lw=lw_main, label='Avg. AUC: %0.3f'% auc_mean)

    
    for i in range(fprs.shape[0]):
        ax.plot(fprs[i], tprs[i], lw=lw_sub, color='darkturquoise', alpha=0.4,\
                 label=f'Seed {i} AUC: %0.3f'% auc_scores_list[i])

    # add standard deviation of auc_scores to legend
    auc_scores_list_std = np.array(auc_scores_list).std()
    ax.plot([0], [0], label = f'AUC std.: %0.3f'% auc_scores_list_std, alpha = 0)
    
    ax.plot([0, 1], [0, 1], color='navy', lw=lw_main-1, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    #find closest to current fpr/tpr on avg_roc using squared distance
    fpr_closest, tpr_closest = min_distance(fprs_mean, tprs_mean, avg_fpr, avg_tpr)
    ax.plot(fpr_closest, tpr_closest, 'go', label = "0.5 thresh. fpr/tpr")

    ax.set_title(f'{type_name}', fontsize = 15, weight='bold') 
    ax.legend(loc="lower right")
    return ax



# Gets all statistics for each seed and makes lists of them. Then prints and draws ROC
def get_all_stats(run, substructs, name, ax, rand_seeds = [3, 4, 8, 9, 27]):
    base_path = os.path.join(run, substructs)
    matrices, accs, auc_scores, balanced_accs, precs, recs, f1s, mccs, specs = get_empty_list(9)
    fpr_point5, tpr_point5 = get_empty_list(2)
    
    for j, seed in enumerate(rand_seeds):
        filename = os.path.join(base_path, name+str(seed))
        with open(filename, "rb") as fil:
            g = pickle.load(fil)
        
        # get lists
        true = np.array(g[0])
        pre = np.array(g[1])
        
        #get confusion matrix
        pres = (pre>0.5).astype(int)
        x = confusion_matrix(true, pres)
        matrices.append(x)
        
        # get accs
        acc = np.diag(x).sum()/x.sum()
        accs.append(acc)
        balanced_acc = (x[0,0]/x[0].sum() + x[1,1]/x[1].sum())/2
        balanced_accs.append(balanced_acc)
        
        
        # get ROC curves
        fpr, tpr, thresholds = roc_curve(true, pre, drop_intermediate = False)
        if j == 0 :
            fprs = fpr
            tprs = tpr
        else:
            fprs = np.vstack((fprs, fpr))
            tprs = np.vstack((tprs, tpr))
            
            
        # get AUCs
        auc_score = roc_auc_score(true, pre)
        auc_scores.append(auc_score)
        
        # get prec, rec, f1
        prec, rec, f1, _ = precision_recall_fscore_support(true, pres, average = "binary")
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)
        
        # get specificity
        spec = x[0][0]/x[0].sum()
        specs.append(spec)
        
        # get mcc
        mcc = matthews_corrcoef(true, pres)
        mccs.append(mcc)
            
    print(substructs, name)
    do_printing(matrices, accs, balanced_accs, precs, recs, f1s, mccs, specs, auc_scores) 
    print()
    print()
    type_name = name.split('_')[0]
    return draw_ROC_curves(fprs, tprs, auc_scores, ax, type_name, recs, specs)


"""
Final function that integrates all of the above
"""
# prints stats and save ROC images for statistics 
def examine(run, subs, subs_names, types, image_filename, square = False, optional_titles = None):
    if square:
        fig, axes = plt.subplots(len(subs),len(types), figsize=(10, 8), sharey = True)
    else:
        fig, axes = plt.subplots(len(subs),len(types), figsize=(13, 8.5), sharey = True)

    for i, sub in enumerate(subs):
        for j, type_ in enumerate(types):
            get_all_stats(run, sub, type_, axes[i, j])
            if j > 0:
                axes[i, j].tick_params(left = False)
            if i > 0:
                axes[i, j].set_title(None) 
            if i == 0:
                axes[i, j].tick_params(bottom = False)
            if j == 0:
                axes[i,j].set_ylabel(subs_names[i], fontsize = 14)

    if optional_titles:
        for i, name in enumerate(optional_titles):
            axes[0, i].set_title(name, fontsize = 15, weight='bold') 
    fig.supxlabel('False Positive Rate', fontsize = 16, weight='bold')
    fig.supylabel('True Positive Rate', fontsize = 16, x = 0, weight='bold')
    fig.tight_layout()
    for ax in fig.get_axes():
        ax.label_outer()
    fig.savefig("./images/"+image_filename, dpi = 500)
    

# same as examine but for a smaller selection of plos
def examine_1_type(run, subs, subs_names, types, plot_names, image_filename, rand_seeds = [3, 4, 8, 9]):
    fig, axes = plt.subplots(len(subs),len(types), figsize=(10, 5), sharey = True)
    sub = subs[0]
    for j, type_ in enumerate(types):
        get_all_stats(run, sub, type_, axes[j], rand_seeds)
        if j > 0:
            axes[j].tick_params(left = False)
        axes[j].set_title(plot_names[j]) 

    fig.supxlabel('False Positive Rate', fontsize = 16, weight='bold')
    fig.supylabel('True Positive Rate', fontsize = 16, x = 0, weight='bold')
    fig.tight_layout()
    for ax in fig.get_axes():
        ax.label_outer()
    fig.savefig("./images/"+image_filename, dpi = 500)