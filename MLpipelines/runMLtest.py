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
from tqdm import tqdm

# Define settings for the experiment 
DATA_DIR = "/ritter/share/data/IMAGEN"
H5_FILES_PATH = "/ritter/share/data/IMAGEN/h5files"
## Permutation tests
# Total number of permutation tests to run. Set to 0 to not perform any permutations. 
N_PERMUTATIONS = 0
PARALLELIZE = False # within each MLPipeline trial, do you want to parallelize the permutation test runs too?
# if set to true it will run 1 trial with no parallel jobs and enables debug msgs
DEBUG = False
    
# directories of all 3 timepoints
tp_dirs = ['newlbls-clean-bl-espad-fu3-19a-binge-n620/*/',
           'newlbls-clean-fu2-espad-fu3-19a-binge-n634/*/', 
           'newlbls-clean-fu3-espad-fu3-19a-binge-n650/*/'] #todo site# :: 'across_sites/lbls-bl-*'
HOLDOUT_DIR = "/ritter/share/data/IMAGEN/h5files/newholdout-clean-{}*{}*.h5" #todo site# newholdout:: h5files/holdout-
SAVE_PATH = "results/holdout-alltp-clean_run.csv" #todo site# #holdout_results :: holdout_results_sites 
EXTRA_INFERENCE_DIR = False# "/ritter/share/data/IMAGEN/h5files/mediumextras-{}*{}*.h5"

if DEBUG:
    if N_PERMUTATIONS > 2: N_PERMUTATIONS = 2
    PARALLELIZE = False
    tp_dirs=[tp_dirs[0]]
    EXTRA_INFERENCE_DIR = False
    
    
##################################################################################################################
if __name__ == "__main__": 
    
    warnings.simplefilter("ignore", UserWarning) ## ignore the warning XGB is throwning
    ##### Load all run.csv files from each dir  #####
    df_runs = []
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
        tp = each_dir.lower().split('-')[2]  # hack: todo make it more automatic
        assert tp in ["bl", "fu2", "fu3"]
        run["tp"] = tp
        df_runs.extend([run])

    df_runs = pd.concat(df_runs).reset_index(drop=True)
    print("running inference  on the holdout set with {} models (3 tps X 4 models X 7 folds) with {} permutation tests in each".format(len(df_runs), N_PERMUTATIONS))

    # get the hyperparameters values to do a gridsearchCV for
    model_grids = { model:
            { hparam: df_runs[hparam].dropna().unique().tolist() 
             for hparam in df_runs.filter(like=f"model_{model}__")} 
            for model in ['LR','SVM-lin','SVM-rbf','GB']}
    
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


    data_reload_flag = ''

    for k, row in tqdm(df_runs.iterrows()):
        # if debug mode then only run 1 in 3 experiments at random
        if DEBUG and k and np.random.choice([True, True, False]):  continue

        if data_reload_flag != row.path: # reload the data and MLpipeline only if the data to be used changed
            data_reload_flag = row.path

            # 1) load MLpipeline class (with a random seed)
            m = MLpipeline(PARALLELIZE, 
                           random_state=np.random.randint(100000), 
                           debug=DEBUG)

            # 2) load the training data    
            h5_path = join(H5_FILES_PATH, "{}.h5".format(row.path.split('/')[-2]))
            print(f"Training data path: \t\t{h5_path}")
            m.load_data(h5_path, y=row.o, confs=['sex', 'site'], group_confs=True)

            # 3) load the holdout data and simulate the MLpipeline.train_test_split() func
            y_name_h5 = slugify(row.o)
            if y_name_h5 == "audit": y_name_h5+="-total"
            test_h5_path = sorted(glob(HOLDOUT_DIR.format(row.tp, y_name_h5))) 
            assert ((len(test_h5_path)==1) or ("binge" in y_name_h5)), "for label {}, multiple test set files found: {}".format(slugify(y_name), test_h5_path)

            h5_path_holdout = test_h5_path[0]
            print(f"Testing data path: \t\t{h5_path_holdout}")
            test_data = h5py.File(h5_path_holdout, "r")  

            m.X_test = test_data['X'][()]
            m.y_test = test_data[row.o][()]  
            m.sub_ids_test = test_data['i'][()]
            for c in m.confs:
                if c != 'group': 
                    m.confs_test[c] = test_data[c][()]
                    # manually redo confs grouping
                    v = test_data[c][()]
                    if "group" not in m.confs_test:
                        m.confs_test["group"] = v
                    else:
                        m.confs_test["group"] = v + 100*m.confs_test["group"]

            m.n_samples_tv = len(m.y)
            m.n_samples_test = len(m.y_test)

            m.print_data_size()

        # 3) Load the trained model
        model_path = glob(f"{row.path}/{row.model}*{row.trial}*.model")[0]
        print(f"ML model (retraining): \t{model_path}")

        model = load(model_path)

        # 4) run the retraining & hyperparameter tuning and generate report
        report = m.run(model, grid=model_grids[row.model], 
                       n_splits=5, conf_corr_params={"conf_corr_cb__groups": m.confs["group"]}, 
                       permute=N_PERMUTATIONS, verbose=1)

        print("train_score: {:0.2f}% \t valid_score: {:0.2f}% \t holdout_score: {:0.2f}".format(
         report['train_score']*100, report['valid_score']*100, report['test_score']*100))

        # store the results in the same dataframe dfIrun
        for key, val in report.items():
            # rename the report columns to have 'holdout'
            new_key = 'holdout_'+ key if 'test' not in key else key.replace('test','holdout')
            df_runs.at[k, new_key]  = val


        if EXTRA_INFERENCE_DIR:
            print("running extra inference also on some (unlabeled) subjects at:\n", EXTRA_INFERENCE_DIR)

            extra_h5_path = sorted(glob(EXTRA_INFERENCE_DIR.format(row.tp, y_name_h5))) 
            assert ((len(extra_h5_path)==1) or ("binge" in y_name_h5)), "for label {}, multiple test set files found: {}".format(slugify(row.o), extra_h5_path)

            extra_h5_path = extra_h5_path[0]
            extra_data = h5py.File(extra_h5_path, "r")
            df_runs.at[k, "holdout_ids_extra"]   = extra_data['i'][()].tolist()

            X_extra = extra_data['X'][()]  
            preds = np.around(model.predict_proba(X_extra), decimals=4)
            df_runs.at[k, "holdout_probs_extra"]  = preds.tolist()

    df_runs.to_csv(SAVE_PATH) 

    # print the total runtime
    runtime=str(datetime.now()-start_time).split(".")[0]
    print("TOTAL RUNTIME: {} secs".format(runtime))