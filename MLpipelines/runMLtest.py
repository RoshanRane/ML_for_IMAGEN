import os, sys
from glob import glob
from os.path import join, isfile
import numpy as np
import pandas as pd

import time
from datetime import datetime
from copy import deepcopy

from sklearn.metrics import balanced_accuracy_score, roc_auc_score, get_scorer, make_scorer, confusion_matrix
from sklearn.feature_selection import SelectKBest, VarianceThreshold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import ParameterGrid, StratifiedKFold

from imblearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB

from joblib import Parallel, delayed, dump, load

# Local imports
from MLpipeline import *
from confounds import *
from slugify import slugify
import warnings
warnings.simplefilter("ignore", UserWarning) ## ignore the warning XGB is throwning

# Define settings for the experiment 
DATA_DIR = "/ritter/share/data/IMAGEN"

## Permutation tests
# Total number of permutation tests to run. Set to 0 to not perform any permutations. 
N_PERMUTATIONS = 0
N_JOBS = 1 # parallel jobs
PARALLELIZE = False # within each MLPipeline trial, do you want to parallelize the permutation test runs too?
# if set to true it will run 1 trial with no parallel jobs and enables debug msgs
DEBUG = False
    
if DEBUG:
    if N_PERMUTATIONS > 5:
        N_PERMUTATIONS = 5
    N_JOBS = 1 
    PARALLELIZE = False
    
dirs = ['newlblst1w-bl-*', 'newlblst1w-fu2-*', 'newlblst1w-fu3-*'] #todo site# :: 'across_sites/lbls-bl-*'
HOLDOUT_DIR = "/ritter/share/data/IMAGEN/h5files/newholdoutt1w-{}*{}*.h5" #todo site# newholdout:: h5files/holdout-
SAVE_PATH = "holdout_results_dummy.csv" #todo site# #holdout_results :: holdout_results_sites 

# sort_by = "roc_auc"  #"roc_auc" #
best_results = pd.DataFrame()

for i, each_dir in enumerate(dirs):
    
    results = pd.DataFrame()
    for f in glob(f"results/{each_dir}/*/run.csv"): 
        run = pd.read_csv(f)
        run = run[~(run.o_is_conf) & ~(run.i_is_conf) & (run.technique=='cb')]
        run["path"] = f.replace("/run.csv", "")
        results = pd.concat([results, run])

    # results = results.sort_values("test_score", ascending=False)
#     results = results.sort_values(sort_by, ascending=False)#.iloc[:10]
#     hyper_cols = results.filter(like="model_").columns
#     results = results.filter(items=["o", "model", "trial", "test_score", "valid_score", "train_score", "roc_auc", *hyper_cols, "path"])
    for tp in ["-bl", "-fu2", "-fu3"]:
        if tp in each_dir:
            results["tp"] = tp[1:]
            break
    best_results = pd.concat([best_results, results])
    
    
best_results = best_results.reset_index(drop=True)

##########################################################
best = []
for tp, grp in best_results.groupby("tp"):
#     best = grp.sort_values(["holdout_score"], ascending=False)#.iloc[:3]
    best_idx = grp[grp["model"].isin(["SVM-rbf", "GB"]) & grp["o"].isin(["Binge"])].index.to_list()#.sort_values(["test_score"], ascending=False).iloc[:3]   
    best.extend(best_idx)
    
best_results = best_results.loc[best].reset_index(drop=True)
print("running {} independent inference with {} permutation tests in each".format(len(best_results), N_PERMUTATIONS))
##########################################################

def run_model(X, y, X_test, y_test, estimator, fit_params={}, permute_X=False):
    
    if permute_X:
        X = MLpipeline._shuffle(X)
        model["conf_corr_cb"].random_state = None
        
    # retrain on entire data
    estimator = estimator.fit(X, y, **fit_params)
    
    test_score = make_scorer(balanced_accuracy_score)(estimator, X_test, y_test)  
    # Calculate AUC if label is binary       
    roc_auc = get_scorer("roc_auc")(estimator, X_test, y_test)  
    
    return test_score, roc_auc

best_results["holdout_score"] = np.nan
best_results["holdout_roc_auc"] = np.nan
best_results["permuted_holdout_score"] = np.nan
best_results["permuted_holdout_score"] = best_results["permuted_holdout_score"].astype('object')
best_results["permuted_holdout_roc_auc"] = np.nan
best_results["permuted_holdout_roc_auc"] = best_results["permuted_holdout_roc_auc"].astype('object')

print("========================================")
start_time = datetime.now()
print("time: ", start_time)
    
for k, row in best_results.iterrows():
    
    tp = row.tp
    print(f"Testing ML model from: {row.path}")
    
    # load the trained model
    model = load(f"{row.path}/{row.model}_{row.trial}.model")
    
    # load the training data    
    h5_path = "/ritter/share/data/IMAGEN/h5files/{}.h5".format(row.path.split('/')[-2])
    data = h5py.File(h5_path, "r")
    X = data['X'][()]
    y_name = data.attrs['labels'][0]
    y = data[y_name][()]    
    confs = {}
    for c in data.attrs['confs']:
        confs[c] = np.array(data[c]) 
        
    # load the holdout data 
    y_name_h5 = slugify(y_name)
    if y_name_h5 == "audit": y_name_h5+="-total"
        
    test_h5_path = sorted(glob(HOLDOUT_DIR.format(tp, y_name_h5))) 
    assert ((len(test_h5_path)==1) or ("binge" in y_name_h5)), "for label {}, multiple test set files found: {}".format(slugify(y_name), test_h5_path)
    h5_path_holdout = test_h5_path[0]
    print(f"Testing on:\t {h5_path_holdout}")
    test_data = h5py.File(h5_path_holdout, "r")
    
    X_test = test_data['X'][()]
    y_test = test_data[y_name][()]    
#     print("n(train_data) = {}\t n(test_data)={} \t n(features) = {}".format(len(X), len(X_test), X.shape[-1]))
    
    # prepare confound-correction params
    confs_grouped = np.array([])
    for c, v in confs.items():               
        if confs_grouped.size == 0:
            confs_grouped = v
        else:
            confs_grouped = 100*confs_grouped + v
            
    fit_params={"conf_corr_cb__groups": confs_grouped}
    # run independent inference
    test_score, roc_auc = run_model(X, y, X_test, y_test, 
                                    model, fit_params)
    
    best_results.loc[k, "holdout_score"]   = test_score    
    best_results.loc[k, "holdout_roc_auc"] = roc_auc  
    
    print("training accuracy: {:0.2f}% \t holdout accuracy: {:0.2f}% \t holdout AUC ROC: {:0.2f}%".format(
     make_scorer(balanced_accuracy_score)(model, X, y)*100, test_score*100, roc_auc*100))
    
    # permutation tests
    with Parallel(n_jobs=N_JOBS) as parallel:
        # run parallel jobs on all cores at once
        pt_scores = parallel(
                        delayed(run_model)
                                (X, y, X_test, y_test, model, fit_params, permute_X=True)
                        for _ in range(N_PERMUTATIONS))
            
    pt_scores  = np.array(pt_scores)           
    print(pt_scores[:,0].tolist())
    best_results.at[k, "permuted_holdout_score"]   = pt_scores[:,0].tolist()
    best_results.at[k, "permuted_holdout_roc_auc"] = pt_scores[:,1].tolist()
                
best_results.to_csv(SAVE_PATH) 

# print the total runtime
runtime=str(datetime.now()-start_time).split(".")[0]
print("TOTAL RUNTIME: {} secs".format(runtime))

