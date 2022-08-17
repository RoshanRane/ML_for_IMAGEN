import os, sys
from glob import glob
from os.path import join, isfile
import numpy as np
import pandas as pd
import h5py 
import time
from datetime import datetime
from copy import deepcopy

from sklearn.metrics import balanced_accuracy_score, roc_auc_score, get_scorer, make_scorer

from joblib import Parallel, delayed, dump, load

# Local imports
from imblearn.pipeline import Pipeline
from MLpipeline import *
from confounds import *
# from slugify import slugify
import warnings
from tqdm import tqdm

### Import the same model pipelines as used in the explorative stage: runMLpipelines file
from runMLpipelines import ML_MODELS 

# Define settings for the experiment 
DATA_DIR = "/ritter/share/data/IMAGEN"
H5_DIR = "/ritter/share/data/IMAGEN/h5files/"
    
# directories with the run results to rerun on holdout data
PERMUTED_H5_DIR = "/ritter/roshan/workspace/ML_for_IMAGEN/post_hoc_milena/h5_permuted_confs/"
PERMUTATION_H5 = [
    # h5cat, training data, holdout data  
    ("bl",      
     PERMUTED_H5_DIR+"permutedconfs10000-newlbls-clean-bl-espad-fu3-19a-binge-n620.h5", 
     H5_DIR+"posthoc-cc2-holdout-h5bl.h5"),
    ("causal1", 
     PERMUTED_H5_DIR+"permutedconfs10000-newlbls-clean-bl-espad-fu3-19a-binge-causal-onset1-n565.h5", 
     H5_DIR+"posthoc-cc2-holdout-h5causal1.h5"),
    ("causal0", 
     PERMUTED_H5_DIR+"permutedconfs10000-newlbls-clean-bl-espad-fu3-19a-binge-causal-onset0-n477.h5", 
     H5_DIR+"posthoc-cc2-holdout-h5causal0.h5"),
]

SAVE_PATH = "results/holdout-posthoc-cc2-permuted_run.csv" 

# Permutation tests
# Total number of permutation tests to run. Set to 0 to not perform any permutations. 
N_PERMUTATIONS = 1000 # 1000
MODEL = 'SVM-rbf' # ['LR','SVM-lin','SVM-rbf','GB']
PARALLELIZE = True # within each MLPipeline trial, do you want to parallelize the permutation test runs too?
# if set to true it will run 1 trial with no parallel jobs and enables debug msgs
DEBUG = False
RAND_STATE=None

if DEBUG:
    if N_PERMUTATIONS > 2: N_PERMUTATIONS = 2
    PARALLELIZE = False
    RAND_STATE = 108
    # PERMUTATION_H5 = [PERMUTATION_H5[0]]
    
    
##################################################################################################################
if __name__ == "__main__": 
    
    df_runs = pd.DataFrame()
    np.random.seed(RAND_STATE)
    ##### STEP 1 : Collect all run.csv file results from provided settings (exp_run_dirs, HOLDOUT_DIR) into a neat table  #####
    # if MODEL is not configured then use all models by default
    assert MODEL in ['LR','SVM-lin','SVM-rbf','GB'], f"invalid models provided in MODEL={MODEL}.\
    Allowed values are ['LR','SVM-lin','SVM-rbf','GB']"
        
    print("========================================")
    start_time = datetime.now()
    print("time: ", start_time)
    
    MODEL_PIPES = {pipe.steps[-1][0].replace('model_', ''):pipe for pipe, grid in ML_MODELS}
    # simulate the oversampling Confound control
    cb = CounterBalance(oversample=True, random_state=RAND_STATE, debug=False)
    {pipe.steps.insert(-1, ("conf_corr_cb", cb))  for name, pipe in MODEL_PIPES.items()}
    MODEL_GRIDS = {pipe.steps[-1][0].replace('model_', ''):grid for pipe, grid in ML_MODELS}
    
    k=0 # df_runs row idx
    for i, (cat, train_h5, hold_h5) in enumerate(PERMUTATION_H5):
        
        print(f"running '{cat}': \n",'-'*30)
        if DEBUG: print(f"Training data path:  {train_h5}")
        # load all data into a dict
        train_data = {}
        rand_confs = []
        with h5py.File(train_h5, 'r') as train_f:
            for key,val in train_f.items():
                train_data.update({key:np.array(val)})
                if 'dummy_' in key: rand_confs.append(key)
        
        # load all holdout data into a dict
        holdout_data = {}
        with h5py.File(hold_h5, 'r') as hold_f:
            for key,val in hold_f.items():
                if key in train_data.keys():
                    holdout_data.update({key: np.array(val)})
            # create random confs for holdout too  
            for rand_conf in rand_confs:
                holdout_data.update({rand_conf: np.random.randint(0,2, size=len(holdout_data['i']))})
                
        
        if DEBUG: 
            print(f"Training data keys: {[k for k in train_data.keys() if 'dummy_' not in k]} \
with n={len([k for k in train_data.keys() if 'dummy_' in k])} dummy confs")
            print(f"Holdout data keys: {[k for k in holdout_data.keys() if 'dummy_' not in k]} \
with n={len([k for k in train_data.keys() if 'dummy_' in k])} dummy confs")

        for j in tqdm(range(N_PERMUTATIONS)):
            k += 1
            conf = rand_confs[j]
            
            df_runs.at[k, 'h5cat']=cat 
            df_runs.at[k, 'train_h5']=train_h5 
            df_runs.at[k, 'hold_h5']=hold_h5 
            df_runs.at[k, 'conf']=conf
            
            # 1) load MLpipeline class (with a random seed)
            m = MLpipeline(PARALLELIZE, 
                           random_state=RAND_STATE, #, 
                           debug=DEBUG)
            # 2) load the training data    
            m.load_data(train_data, y='Binge', 
                        confs=['sex', 'site', conf], 
                        group_confs=True)

            # 3) load the holdout data
            #  and perform the MLpipeline.train_test_split() func explicitly 
            m.X_test = holdout_data['X']
            m.y_test = holdout_data['Binge'] 
            m.sub_ids_test = holdout_data['i']

            for c in m.confs:
                if c != 'group': 
                    m.confs_test[c] = holdout_data[c][()]
                    # manually redo confs grouping
                    v = holdout_data[c][()]
                    if "group" not in m.confs_test:
                        m.confs_test["group"] = v
                    else:
                        m.confs_test["group"] = v + 100*m.confs_test["group"]

            m.n_samples_tv = len(m.y)
            m.n_samples_test = len(m.y_test)

            if DEBUG: m.print_data_size()

            # 4) re-run the training 
            model = MODEL_PIPES[MODEL]
            # when output is not the label 'y', perform counterbalancing across the label 'y'
            # for 'c' using the variable 'groups' in the cb function
            conf_corr_params = {"conf_corr_cb__groups": m.confs["group"]} 
            if DEBUG: print("model pipe = ", model)

            # get the hyperparameters values from previous run
            grid = MODEL_GRIDS[MODEL]

            report = m.run(model, grid=grid, 
                           n_splits=5, conf_corr_params=conf_corr_params, 
                           permute=0, verbose=int(DEBUG))

            print("train_score: {:0.2f}% \t valid_score: {:0.2f}% \t holdout_score: {:0.2f}".format(
             report['train_score']*100, report['valid_score']*100, report['test_score']*100))
            
            # store the results in the same dataframe dfIrun
            for key, val in report.items():
                # rename the report columns to have 'holdout'
                new_key = 'holdout_'+ key if 'test' not in key else key.replace('test','holdout')
                if not isinstance(val, (np.ndarray, list)): 
                    df_runs.at[k, new_key]  = val

            if DEBUG: 
                if j>50: break # in debug mode run on maximum 10 samples

        # iteratively store results every 50 steps
            if k%50 == 0: 
                df_runs.to_csv(SAVE_PATH) 
                runtime=str(datetime.now()-start_time).split(".")[0]
                print("RUNTIME: {} secs".format(runtime))
    
    df_runs.to_csv(SAVE_PATH) 

    # print the total runtime
    runtime=str(datetime.now()-start_time).split(".")[0]
    print("TOTAL RUNTIME: {} secs".format(runtime))