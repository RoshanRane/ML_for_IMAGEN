import os, sys
from glob import glob
from os.path import join, isfile
import numpy as np
import pandas as pd

import time
from datetime import datetime
from copy import deepcopy

from sklearn.metrics import balanced_accuracy_score, roc_auc_score, get_scorer, make_scorer

from joblib import Parallel, delayed, dump, load

# Local imports
from MLpipeline import *
from confounds import *
from slugify import slugify
import warnings
warnings.simplefilter("ignore", UserWarning) ## ignore the warning XGB is throwning

# Define settings for the experiment 
DATA_DIR = "/ritter/share/data/IMAGEN"
H5_FILES_PATH = "/ritter/share/data/IMAGEN/h5files"
## Permutation tests
# Total number of permutation tests to run. Set to 0 to not perform any permutations. 
N_PERMUTATIONS = 0
N_JOBS = 1 # parallel jobs
PARALLELIZE = False # within each MLPipeline trial, do you want to parallelize the permutation test runs too?
# if set to true it will run 1 trial with no parallel jobs and enables debug msgs
DEBUG = False
    
# directories of all 3 timepoints
tp_dirs = ['newlbls-bl-espad-fu3-19a-binge-n620/20210618-1632',
           'newlbls-fu2-espad-fu3-19a-binge-n634/20210618-1701', 
           'newlbls-fu3-espad-fu3-19a-binge-n650/20210618-1730'] #todo site# :: 'across_sites/lbls-bl-*'
HOLDOUT_DIR = "/ritter/share/data/IMAGEN/h5files/newholdout-{}*{}*.h5" #todo site# newholdout:: h5files/holdout-
SAVE_PATH = "results/holdout_all-tp-extra_run.csv" #todo site# #holdout_results :: holdout_results_sites 
EXTRA_INFERENCE_DIR = "/ritter/share/data/IMAGEN/h5files/mediumextras-{}*{}*.h5"
    
if DEBUG:
    if N_PERMUTATIONS > 5: N_PERMUTATIONS = 5
    N_JOBS = 1 
    PARALLELIZE = False
    tp_dirs=[tp_dirs[0]]

# Define a function for running training and test
def run_model(X, y, X_test, y_test, estimator, fit_params={}, permute_X=False):
    if permute_X:
        X = MLpipeline._shuffle(X)
        estimator["conf_corr_cb"].random_state = None
    # retrain on entire data
    estimator = estimator.fit(X, y, **fit_params)
    
    # calculate balanced accuracy and AUC-ROC on test data
    test_score = make_scorer(balanced_accuracy_score)(estimator, X_test, y_test)        
    roc_auc = get_scorer("roc_auc")(estimator, X_test, y_test)  
    predicted_probs = np.around(estimator.predict_proba(X_test), decimals=4)
    return estimator, test_score, roc_auc, predicted_probs


if __name__ == "__main__": 
    
    # 1) Load all run.csv files from each dir
    runs = []
    for i, each_dir in enumerate(tp_dirs):
        # first, collect all run.csv results together
        f = glob(f"results/{each_dir}/run.csv")
        assert len(f)==1, f"multiple result folders found with {each_dir} = {f}"
        f = f[0]
        run = pd.read_csv(f)
        # select only 'cb' controlled results and drop the confound-related rows
        run = run[~(run.o_is_conf) & ~(run.i_is_conf) & (run.technique=='cb')]
        # add a column about the source h5 file
        run["path"] = f.replace("/run.csv", '')
        # add a column about the tp in the source h5 file  
        tp = each_dir.lower().split('-')[1]
        assert tp in ["bl", "fu2", "fu3"]
        run["tp"] = tp
        runs.extend([run])

    runs = pd.concat(runs).reset_index(drop=True)
#     print(runs)

    # 2) Select the best models based on some criteria
    # best = []
    # for tp, grp in runs.groupby("tp"):
    # #     best = grp.sort_values(["holdout_score"], ascending=False)#.iloc[:3] # criteria: best models
    #     best_idx = grp[grp["model"].isin(["SVM-rbf", "GB"]) & grp["o"].isin(["Binge"])].index.to_list()#.sort_values(["test_score"], ascending=False).iloc[:3]   
    #     best.extend(best_idx)
    # runs = runs.loc[best].reset_index(drop=True)
    print("running inference  on the holdout set with {} models (3 tps X 4 models X 7 folds) with {} permutation tests in each".format(len(runs), N_PERMUTATIONS))
    
    # 3) Initialize new columns to store results from holdout
    runs["holdout_score"] = np.nan
    runs["holdout_roc_auc"] = np.nan
    runs["holdout_ids"]= np.nan
    runs["holdout_lbls"]= np.nan
    runs["holdout_probs"]= np.nan
    runs["permuted_holdout_score"] = np.nan
    runs["permuted_holdout_roc_auc"] = np.nan
    runs["holdout_ids_extra"] = np.nan
    runs["holdout_probs_extra"] = np.nan
    for c in ["holdout_ids", "holdout_lbls","holdout_probs",
              "permuted_holdout_score", "permuted_holdout_roc_auc",
              "holdout_ids_extra", "holdout_probs_extra"]:
        runs[c] = runs[c].astype('object')

    print("========================================")
    start_time = datetime.now()
    print("time: ", start_time)

    for k, row in runs.iterrows():
        
        tp = row.tp
        print(f"Retraining ML model: \t{row.path}")

        # load the trained model
        model = load(f"{row.path}/{row.model}_{row.trial}.model")

        # load the training data    
        h5_path = join(H5_FILES_PATH, "{}.h5".format(row.path.split('/')[-2]))
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
        print(f"Testing on: \t\t{h5_path_holdout}")
        test_data = h5py.File(h5_path_holdout, "r")

        X_test = test_data['X'][()]
        y_test = test_data[y_name][()]    
#         print("n(train_data) = {}\t n(test_data)={} \t n(features) = {}".format(len(X), len(X_test), X.shape[-1]))
        runs.at[k, "holdout_ids"]   = test_data['i'][()].tolist()
        runs.at[k, "holdout_lbls"]  = y_test.tolist()
        
        # prepare confound-correction params
        confs_grouped = np.array([])
        for c, v in confs.items():               
            if confs_grouped.size == 0:
                confs_grouped = v
            else:
                confs_grouped = 100*confs_grouped + v

        fit_params={"conf_corr_cb__groups": confs_grouped}
        
        # run independent inference
        model, test_score, roc_auc, preds = run_model(X, y, X_test, y_test, 
                                        model, fit_params)
        
        runs.at[k, "holdout_probs"]  = preds.tolist()
        runs.loc[k, "holdout_score"]   = test_score    
        runs.loc[k, "holdout_roc_auc"] = roc_auc  

        print("training accuracy: {:0.2f}% \t holdout accuracy: {:0.2f}% \t holdout AUC ROC: {:0.2f}%".format(
         make_scorer(balanced_accuracy_score)(model, X, y)*100, test_score*100, roc_auc*100))
        
        if EXTRA_INFERENCE_DIR:
            print("running extra inference also on some (unlabeled) subjects at:\n", EXTRA_INFERENCE_DIR)
        
            extra_h5_path = sorted(glob(EXTRA_INFERENCE_DIR.format(tp, y_name_h5))) 
            assert ((len(extra_h5_path)==1) or ("binge" in y_name_h5)), "for label {}, multiple test set files found: {}".format(slugify(y_name), extra_h5_path)

            extra_h5_path = extra_h5_path[0]
            extra_data = h5py.File(extra_h5_path, "r")
    #         print("n(train_data) = {}\t n(test_data)={} \t n(features) = {}".format(len(X), len(X_test), X.shape[-1]))
            runs.at[k, "holdout_ids_extra"]   = extra_data['i'][()].tolist()

            X_extra = extra_data['X'][()]  
            preds = np.around(model.predict_proba(X_extra), decimals=4)
            runs.at[k, "holdout_probs_extra"]  = preds.tolist()
    
        # permutation tests
        if N_PERMUTATIONS:
            with Parallel(n_jobs=N_JOBS) as parallel:
                # run parallel jobs on all cores at once
                pt_scores = parallel(
                                delayed(run_model)
                                        (X, y, X_test, y_test, model, fit_params, permute_X=True)
                                for _ in range(N_PERMUTATIONS))

            pt_scores  = np.array(pt_scores)           
#             print(pt_scores[:,0].tolist())
            runs.at[k, "permuted_holdout_score"]   = pt_scores[:,0].tolist()
            runs.at[k, "permuted_holdout_roc_auc"] = pt_scores[:,1].tolist()

    runs.to_csv(SAVE_PATH) 

    # print the total runtime
    runtime=str(datetime.now()-start_time).split(".")[0]
    print("TOTAL RUNTIME: {} secs".format(runtime))
