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
H5_DIR = "/ritter/share/data/IMAGEN/h5files"
rerun_ver = 'posthoc-cc3' # posthoc-cc3 or posthoc-cc2 or posthoc-cc
# directories with the run results to rerun on holdout data
exp_run_dirs = [
                # f'{rerun_ver}-h5bl-*/',
                f'{rerun_ver}-h5causal1-*/', 
                f'{rerun_ver}-h5causal0-*/',]

HOLDOUT_DIR = [
               # f'{rerun_ver}-holdout-h5bl.h5',
               f'{rerun_ver}-holdout-h5causal0.h5',
               f'{rerun_ver}-holdout-h5causal1.h5',]

SAVE_PATH = f"results/holdout-{rerun_ver}_run.csv" 

# Permutation tests
# Total number of permutation tests to run. Set to 0 to not perform any permutations. 
N_PERMUTATIONS = 0 # 1000
USE_ONLY_MODELS = ['SVM-rbf'] # ['LR','SVM-lin','SVM-rbf','GB']
PARALLELIZE = True # within each MLPipeline trial, do you want to parallelize the permutation test runs too?
# if set to true it will run 1 trial with no parallel jobs and enables debug msgs
DEBUG = False
RAND_STATE=None

if DEBUG:
    if N_PERMUTATIONS > 2: N_PERMUTATIONS = 2
    PARALLELIZE = False
    RAND_STATE = 108
    
    
##################################################################################################################
if __name__ == "__main__": 
    
    ## Step 0 : load all holdout data before-hand as  dict {h5 category: loaded h5 file handler}
    hold_h5_fs = {}
    hold_h5s_confs = {}
    for hold_h5 in HOLDOUT_DIR:
        hold_h5cat = hold_h5.split('.')[-2].replace(f'{rerun_ver}-holdout-h5','').replace('/','')
        hold_f=h5py.File(join(H5_DIR, hold_h5), 'r')
        hold_h5_fs.update({hold_h5cat : hold_f})
        hold_h5s_confs.update({hold_h5cat : list(hold_f.attrs['confs'])})
    if DEBUG: print("holdout files:", hold_h5_fs)
        # print("holdout file confounds:", hold_h5s_confs)    
        
    ##### STEP 1 : Collect all run.csv file results from provided settings (exp_run_dirs, HOLDOUT_DIR) into a neat table  #####
    # if USE_ONLY_MODELS is not configured then use all models by default
    if len(USE_ONLY_MODELS)==0: 
        USE_ONLY_MODELS=['LR','SVM-lin','SVM-rbf','GB']
    for m in USE_ONLY_MODELS:
        assert m in ['LR','SVM-lin','SVM-rbf','GB'], f"invalid models provided in USE_ONLY_MODELS={USE_ONLY_MODELS}.\
    Allowed values are ['LR','SVM-lin','SVM-rbf','GB']"
        
    
    df_runs = []
    for _, each_path in enumerate(exp_run_dirs):
        all_runs = glob(f"results/{each_path}/*/run.csv")
        print(f"Collecting all n={len(all_runs)} results (run.csv files) in the dir results/{each_path}/*/")
        for f in all_runs:
            # first, collect all run.csv results together
            run = pd.read_csv(f)
            # select only 'cb' controlled results, drop confound-related analysis and use only the configured models
            run = run[~(run.o_is_conf) & ~(run.i_is_conf) & (run.technique=='cb') & run.model.isin(USE_ONLY_MODELS)]
        
            # add a column about the source h5 file, h5 category ("bl", "causal0", "causal1", "fu2", "fu3")
            results, folder, timeline, runcsv = f.split('/')
            # todo: hacky code below
            h5cat = folder.lower().split('-')[2].replace('h5','')  
            conf = folder.replace(f'{rerun_ver}-h5{h5cat}-', '')
            # print('[D]',folder, h5cat, conf)
            run["h5cat"] = h5cat
            run["conf"] = conf
            run["path"] = folder + '/' + timeline
            
            # sanity checks
            assert results=='results' and runcsv=='run.csv', f'sanity check for path {f} failed. \
Below code might not work as expected anymore..'
            assert h5cat in ["bl", "causal0", "causal1", "fu2", "fu3"]
            if h5cat not in hold_h5_fs.keys():
                print(f"Skipping {folder}: no holdout data. Provided holdout data have h5cats = {list(hold_h5_fs.keys())} but this file has h5cat = {h5cat}")
                continue
            # filter out runcsvs that don't have a corresponding holdout data
            if conf not in hold_h5s_confs[h5cat]:
                print(f"Skipping {folder}: conf '{run['conf']}' not available in holdout data {hold_h5s_confs[h5cat]}")
                continue
                
            df_runs.append(run)

    df_runs = pd.concat(df_runs).reset_index(drop=True)
    print("running inference  on the holdout set with {} models with {} permutation tests in each".format(len(df_runs), N_PERMUTATIONS))
    
    
    ##### Initialize new columns to store results from holdout #####
    for c in ["holdout_score", "holdout_roc_auc",
              "holdout_ids", "holdout_lbls","holdout_probs",
              "permuted_holdout_score", "holdout_permuted_roc_auc",
              "holdout_ids_extra", "holdout_probs_extra"
             ]:
        df_runs[c] = np.nan
        df_runs[c] = df_runs[c].astype('object')

    print("========================================")
    start_time = datetime.now()
    print("time: ", start_time)
    
    
    MODEL_PIPES = {pipe.steps[-1][0].replace('model_', ''):pipe for pipe, grid in ML_MODELS}
    # simulate the oversampling Confound control
    cb = CounterBalance(oversample=True, random_state=RAND_STATE, debug=False)
    {pipe.steps.insert(-1, ("conf_corr_cb", cb))  for name, pipe in MODEL_PIPES.items()}

#     data_reload_flag = ''

    warnings.simplefilter("ignore", UserWarning) ## ignore the warning XGB is throwing
    
    for k, row in tqdm(df_runs.iterrows()):
#         # if debug mode then only run 1 in 3 experiments at random
#         if DEBUG and k and np.random.choice([True, True, False]):  continue

#         if data_reload_flag != row.path: # reload the data and MLpipeline only if the data to be used changed
#             data_reload_flag = row.path

        # 1) load MLpipeline class (with a random seed)
        m = MLpipeline(PARALLELIZE, 
                       random_state=RAND_STATE, #, 
                       debug=DEBUG)

        # 2) load the training data    
        h5_path = join(H5_DIR, "{}.h5".format(row.path.split('/')[-2]))
        conf = row.conf
        if DEBUG: print(f"Training data path: \t {h5_path}")
        m.load_data(h5_path, y=row.o, confs=['sex', 'site', conf], group_confs=True)

        # 3) load the holdout data
        #  and perform the MLpipeline.train_test_split() func explicitly 
        holdout_data = hold_h5_fs[row.h5cat]
        m.X_test = holdout_data['X'][()]
        m.y_test = holdout_data[row.o][()]  
        m.sub_ids_test = holdout_data['i'][()]
        
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
        model = MODEL_PIPES[row.model]
        # when output is not the label 'y', perform counterbalancing across the label 'y'
        # for 'c' using the variable 'groups' in the cb function
        conf_corr_params = {"conf_corr_cb__groups": m.confs["group"]} 
        if DEBUG: print("model pipe = ", model)
        
        # get the hyperparameters values from previous run
        grid = {}
        for hparam, hparam_val in row.filter(like=f"model_{row.model}__").items():
            if not pd.isna(hparam_val):
                grid.update({hparam: [hparam_val]})
        if DEBUG: print("model grid = ", grid)
        
        report = m.run(model, grid=grid, 
                       n_splits=5, conf_corr_params=conf_corr_params, 
                       permute=N_PERMUTATIONS, verbose=1)

        print("train_score: {:0.2f}% \t valid_score: {:0.2f}% \t holdout_score: {:0.2f}".format(
         report['train_score']*100, report['valid_score']*100, report['test_score']*100))

        # store the results in the same dataframe dfIrun
        for key, val in report.items():
            # rename the report columns to have 'holdout'
            new_key = 'holdout_'+ key if 'test' not in key else key.replace('test','holdout')
            df_runs.at[k, new_key]  = val
            
        if DEBUG: 
            if k>50: break # in debug mode run on maximum 10 samples
     
    # iteratively store results every 50 steps
        if k% 50 == 0: 
            df_runs.to_csv(SAVE_PATH) 
    
    df_runs.to_csv(SAVE_PATH) 

    # print the total runtime
    runtime=str(datetime.now()-start_time).split(".")[0]
    print("TOTAL RUNTIME: {} secs".format(runtime))