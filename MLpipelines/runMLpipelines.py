'''
File with configurations to run MLpipeline() class from MLpipeline.py on the IMAGEN data.
This configuration was used for Evert's thesis' experiments and serves as an example for how to 
configure the MLpipeline class and run it.
'''
import os, sys
from glob import glob
from os.path import join, isfile
import numpy as np
import pandas as pd

import time
from datetime import datetime
from copy import copy, deepcopy

from sklearn.metrics import balanced_accuracy_score, roc_auc_score, get_scorer, make_scorer, confusion_matrix
from sklearn.feature_selection import SelectKBest, VarianceThreshold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import ParameterGrid, StratifiedKFold, LeaveOneGroupOut

from imblearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB

from joblib import Parallel, delayed, dump, load
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# Local imports
from MLpipeline import *
from confounds import *

# Define settings for the experiment 
DATA_DIR = "/ritter/share/data/IMAGEN/"
## CV loops
N_OUTER_CV = 7 # number of folds in inner crossvalidation for test score estimation
N_INNER_CV = 5 # number of folds in inner crossvalidation used for hyperparameter tuning
## Optional runs
CONF_CTRL_TECHS = ["baseline-cb", "cb"] # choose from ["baseline", "cb", "cr", "loso"]  ####
RUN_CONFS = True ####
EXCLUDE_IN_RUN_CONFS = ['sex', 'site'] #### 'sex', 'site'

SAVE_MODELS = False # saves the final trained models but only for io=={X-y} and conf_ctrl_tech=='CB' ####
RUN_PBCC = False # run the prediction-based post-prediction conf_ctrl_tech by Dinga et al. 2020
RUN_CHI_SQUARE = False # runs a chi-square analysis between the label and all the confounds (todo: only supports categorical confounds)

## Permutation tests
# Total number of permutation tests to run. Set to 0 to not perform any permutations. 
N_PERMUTATIONS = 0
PERMUTE_ONLY_XY = True
N_JOBS = 2 # parallel jobs ####
PARALLELIZE = False # within each MLPipeline trial, do you want to parallelize the permutation test runs too?
# if set to true it will run 1 trial with no parallel jobs and enables debug msgs
DEBUG = True ####
    
if DEBUG:
    N_OUTER_CV = 2
    N_INNER_CV = 2
    if N_PERMUTATIONS > 2:
        N_PERMUTATIONS = 2
    N_JOBS = 1 
    PARALLELIZE = False
        
# The ML pipelines to run and their corresponding hyperparameter grids as tuples i.e. (pipeline, grid)
ML_MODELS = [
    ( # (pipeline, grid) for Logistic Regression classifier
        Pipeline([
            ("varth", VarianceThreshold()), 
            ("scale", StandardScaler()),
            ("model_LR", LogisticRegression(max_iter=1000))
            ]),
        {"model_LR__C" : [1000, 100, 1.0, 0.001]}, 
    ),
    ( # (pipeline, grid) for linear SVM classifier
        Pipeline([
            ("varth", VarianceThreshold()),
            ("scale", StandardScaler()),
            ("model_SVM-lin", SVC(kernel="linear", max_iter=10000, probability=True))
            ]),
        {"model_SVM-lin__C" : [1000, 100, 1.0, 0.001]},     
    ),
    ( # (pipeline, grid) for SVM classifier with rbf kernel
        Pipeline([
            ("varth", VarianceThreshold()),
            ("scale", StandardScaler()),
            ("model_SVM-rbf", SVC(kernel="rbf", probability=True))
            ]),
        {
            "model_SVM-rbf__C" : [1000, 100, 1.0, 0.001],
            "model_SVM-rbf__gamma" : ['scale', 'auto']
        }
    ),
    ( # (pipeline, grid) for GradientBoosting classifier
        Pipeline([
        ("varth", VarianceThreshold()),
        ("scale", StandardScaler()),
        ("model_GB", XGBClassifier(n_estimators=100, max_depth=5, subsample=1.0, 
                                   use_label_encoder=True,  eval_metric='logloss'))
        ]),
        {
            "model_GB__learning_rate" : [0.05, 0.25],
            # todo dd: very few tuning done as it is too expensive for GB
        } 
    )
]

# Here you can select which HDF5 files you want to include in analysis. 
H5_FILES = [  ####
### main
# '/ritter/share/data/IMAGEN/h5files/posthoc-cc-ctq-denial-sum-n650.h5',
    DATA_DIR + 'h5files/posthoc-cc-pss-pss-tota-n650.h5',
#  '/ritter/share/data/IMAGEN/h5files/newlbls-clean-bl-audit-fu3-audit-freq-audit-quick-n614.h5',
#  '/ritter/share/data/IMAGEN/h5files/newlbls-clean-bl-audit-fu3-audit-total-audit-n687.h5',
#  '/ritter/share/data/IMAGEN/h5files/newlbls-clean-bl-audit-fu3-audit2-amount-n567.h5',
#  '/ritter/share/data/IMAGEN/h5files/newlbls-clean-bl-audit-gm-fine-cluster-audit-growth-n759.h5',
#  '/ritter/share/data/IMAGEN/h5files/newlbls-clean-bl-espad-fu3-19a-binge-n620.h5',
#  '/ritter/share/data/IMAGEN/h5files/newlbls-clean-bl-espad-fu3-29d-onset-15-n654.h5',
#  '/ritter/share/data/IMAGEN/h5files/newlbls-clean-bl-espad-fu3-8b-frequency-n675.h5',
#  '/ritter/share/data/IMAGEN/h5files/newlbls-clean-bl-espad-gm-fine-cluster-binge-growth-n661.h5',
#  '/ritter/share/data/IMAGEN/h5files/newlbls-clean-bl-our-combo-cluster-combined-ours-n566.h5',
#  '/ritter/share/data/IMAGEN/h5files/newlbls-clean-bl-phenotype-phenotype-combined-seo-n740.h5',
#  '/ritter/share/data/IMAGEN/h5files/newlbls-clean-fu2-audit-fu3-audit-freq-audit-quick-n628.h5',
#  '/ritter/share/data/IMAGEN/h5files/newlbls-clean-fu2-audit-fu3-audit-total-audit-n705.h5',
#  '/ritter/share/data/IMAGEN/h5files/newlbls-clean-fu2-audit-fu3-audit2-amount-n572.h5',
#  '/ritter/share/data/IMAGEN/h5files/newlbls-clean-fu2-audit-gm-fine-cluster-audit-growth-n713.h5',
#  '/ritter/share/data/IMAGEN/h5files/newlbls-clean-fu2-espad-fu3-19a-binge-n634.h5',
#  '/ritter/share/data/IMAGEN/h5files/newlbls-clean-fu2-espad-fu3-29d-onset-15-n666.h5',
#  '/ritter/share/data/IMAGEN/h5files/newlbls-clean-fu2-espad-fu3-8b-frequency-n698.h5',
#  '/ritter/share/data/IMAGEN/h5files/newlbls-clean-fu2-espad-gm-fine-cluster-binge-growth-n679.h5',
#  '/ritter/share/data/IMAGEN/h5files/newlbls-clean-fu2-our-combo-cluster-combined-ours-n582.h5',
#  '/ritter/share/data/IMAGEN/h5files/newlbls-clean-fu2-phenotype-phenotype-combined-seo-n782.h5',
#  '/ritter/share/data/IMAGEN/h5files/newlbls-clean-fu3-audit-fu3-audit-freq-audit-quick-n623.h5',
#  '/ritter/share/data/IMAGEN/h5files/newlbls-clean-fu3-audit-fu3-audit-total-audit-n708.h5',
#  '/ritter/share/data/IMAGEN/h5files/newlbls-clean-fu3-audit-fu3-audit2-amount-n585.h5',
#  '/ritter/share/data/IMAGEN/h5files/newlbls-clean-fu3-audit-gm-fine-cluster-audit-growth-n589.h5',
#  '/ritter/share/data/IMAGEN/h5files/newlbls-clean-fu3-espad-fu3-19a-binge-n650.h5',
#  '/ritter/share/data/IMAGEN/h5files/newlbls-clean-fu3-espad-fu3-29d-onset-15-n697.h5',
#  '/ritter/share/data/IMAGEN/h5files/newlbls-clean-fu3-espad-fu3-8b-frequency-n709.h5',
#  '/ritter/share/data/IMAGEN/h5files/newlbls-clean-fu3-espad-gm-fine-cluster-binge-growth-n570.h5',
#  '/ritter/share/data/IMAGEN/h5files/newlbls-clean-fu3-our-combo-cluster-combined-ours-n588.h5',
#  '/ritter/share/data/IMAGEN/h5files/newlbls-clean-fu3-phenotype-phenotype-combined-seo-n589.h5',
]

def conf_corr_run(h5_file, 
                  conf_ctrl_tech, io, model_pipegrid, trial, test_idx,
                  label_name, save_dir, confs, n_inner_cv, run_pbcc,
                  parallelize, n_permutes_per_trial, permute_only_xy, 
                  save_models, debug, random_state=None):
    
    start_time_this_thread = datetime.now()
    conf_ctrl_tech = conf_ctrl_tech.lower()
    inp, out = io
    pipe, grid = deepcopy(model_pipegrid)
    model_name = pipe.steps[-1][0].replace("model_", "")    
    print("--------------------------------------")
    print("Starting a new pipeline with setting:\n conf_ctrl_tech={}, io={}, model={}, outer_cv_trial={}".format(
    conf_ctrl_tech, io, model_name, trial))
    
    m = MLpipeline(parallelize, random_state=random_state, debug=debug)
    # load X, y and confounds
    m.load_data(h5_file, y=label_name, confs=confs, group_confs=True)
    # randomly split data into training and test set
    m.train_test_split(test_idx=test_idx) 
                                 
    ### <START> Special conditions for each confound correction conf_ctrl_tech
    conf_corr_params = {}  
    stratify_by_conf = None
    n_samples_cc = m.n_samples_tv
    # 1) CounterBalancing
    if "cb" in conf_ctrl_tech:
        # Counterbalance for both sex and site, which is "group"
        oversample = True
        if conf_ctrl_tech == "under-cb":
            oversample=False
        elif conf_ctrl_tech == "overunder-cb":
            oversample=None
        else:
            oversample=True
            
        # if 'baseline-cb' then control only for 'sex' and 'site' not for any of the other additional variable/s given            
        if 'baseline' in conf_ctrl_tech:
            # repeat the loading
            m = MLpipeline(parallelize, random_state=random_state, debug=debug)
            m.load_data(h5_file, y=label_name, confs=confs, group_confs=True)
            # correct the 'grouped' confs to only include 'sex' and 'site'
            m.confs["group"] = m.confs['sex'] + 100*m.confs["site"]
            # repeat the splitting
            m.train_test_split(test_idx=test_idx) 
        
        cb = CounterBalance(oversample, random_state=random_state, debug=debug)
        pipe.steps.insert(-1, ("conf_corr_cb", cb))
        conf_corr_params.update({"conf_corr_cb__groups": m.confs["group"]})
        # when output is not the label 'y', still perform counterbalancing across the label 'y'
        # because we care about the effect of 'c' 
        if (out in confs): conf_corr_params.update({"conf_corr_cb__cb_by": m.y}) 
        # calculate info about how CB changes the training sample size
        n_samples_cc = len(cb._get_cb_sampled_idxs(groups=m.confs["group"], cb_by=m.y)) 
        
    # 2) Confound Regression
    elif (conf_ctrl_tech in ["cr"]) and (inp == "X"):
        cr = ConfoundRegressorCategoricalX(debug=debug)
        pipe.steps.insert(-1, ("conf_corr_cr", cr))
        conf_corr_params.update({"conf_corr_cr__groups": m.confs["group"]})
        
    ### <END> Special conditions for each conf_ctrl_conf_ctrl_tech

    if (inp in confs): m.change_input_to_conf(inp, onehot=True) # todo: onehot is hardcoded as confounds are assumed as categorical
    if (out in confs): m.change_output_to_conf(out)
    
    # run pbcc only for X-y
    if ((inp in confs) or (out in confs)): 
        run_pbcc=False
    
    # run permutation for other than X-y experiments?
    if permute_only_xy and ((inp in confs) or (out in confs)):
        n_permutes_per_trial = 0
        
    # Run the actual classification pipeline including the hyperparameter tuning
    run = m.run(pipe, grid, 
                 n_splits=n_inner_cv, 
                 conf_corr_params=conf_corr_params,
                 stratify_by_conf=stratify_by_conf,
                 run_pbcc=run_pbcc,
                 permute=n_permutes_per_trial)
    
    # prepare results
    result = {
        "io" : "{}-{}".format(inp,out),
        "technique" : conf_ctrl_tech,
        "model" : model_name,
        "trial" : trial,
        "n_samples":(m.n_samples_tv + m.n_samples_test),
        "n_samples_cc":(n_samples_cc + m.n_samples_test),
        "i" : inp,
        "o" : out,
        "i_is_conf" : (inp in confs),
        "o_is_conf" : (out in confs),
    }
    # Append results
    result.update(run)    
    
    runtime = int((datetime.now() - start_time_this_thread).total_seconds())
    result.update({"runtime":runtime})
    print("Finished after {}s with test_score = {:.2f}".format(
        str(datetime.now() - start_time_this_thread).split(".")[0], result['test_score']*100))    
    df = pd.DataFrame([result])
    df.to_csv(join(save_dir, f"run_{label_name}_{random_state}.csv" ))
    
    # save models only for X-y experiments with conf_ctrl_tech == CounterBalance
    if save_models and (i not in confs) and (o not in confs) and (conf_ctrl_tech!="baseline"):
        dump(m.estimator, join(save_dir, f"{model_name}_{conf_ctrl_tech}_{trial}.model"))
        
        
#########################################################################################################################

def runMLpipelines(
            h5_files, ML_models, 
            n_outer_cv=5, n_inner_cv=5, 
            conf_ctrl_techs=['baseline'], run_confs=False, exclude_in_run_confs=[],
            n_jobs=1, parallelize=False, 
            n_permutations=0, permute_only_xy=True,
            save_models=False, debug=False, 
            run_chi_square=False, run_pbcc=False):
    
    # The total number of permutations that are run per trial
    n_permutes_per_trial = n_permutations//n_outer_cv
    
    with Parallel(n_jobs=n_jobs) as parallel:
        
        for h5_file in h5_files:

            print("========================================")
            print("Running MLpipeline on file:\n", h5_file)
            start_time = datetime.now()
            print("time: ", start_time)
            # Create the folder in which to save the results
            if debug: 
                os.system("rm -rf results/debug_run 2> /dev/null")
                save_dir = "results/debug_run/{}".format(
                start_time.strftime("%Y%m%d-%H%M"))
            else:
                save_dir = "results/{}/{}".format(
                    os.path.basename(h5_file).replace(".h5",""),
                    start_time.strftime("%Y%m%d-%H%M"))
            if not os.path.isdir(save_dir): os.makedirs(save_dir)
            
            # load the data.h5 file
            data = h5py.File(h5_file, "r")
            data_size = len(data["X"])
            # determine the input-output combinations to run from the h5 file
            conf_names = data.attrs["confs"].tolist()            
            label_names = data.attrs["labels"].tolist()            
            labels = pd.DataFrame({lbl :np.array(data[lbl]) for lbl in label_names})
            assert len(label_names)==1, "multiple labels are not supported\
in imagen_ml repository since the commit 7f5b67e95d605f3218d96199c07e914589a9a581."
            y = label_names[0]
            # prepare the "io"
            io_combinations = [("X", y)]
            if run_confs:
                exclude_in_run_confs = [c.lower() for c in exclude_in_run_confs]
                # skip confound-based analysis if not explicitly requested
                io_combinations.extend([(c , y) for c in conf_names if c.lower() not in exclude_in_run_confs]) # Same analysis approach
                io_combinations.extend([("X", c) for c in conf_names if c.lower() not in exclude_in_run_confs]) # Snoek et al.        
            
            # generate all setting combinations of (1) conf_ctrl_techs, (2) INPUT_OUTPUT combination,
            # (3) MODEL, and (4) n_outer_cv trials so that they can be run in parallel
            settings = []
            for conf_ctrl_tech in conf_ctrl_techs:
                for io in io_combinations:
                    for ML_model in ML_models: # pipe=model_pipeline, grid=hyperparameters
                        # pre-generate the test indicies for the outer CV as they need to run in parallel
                        if conf_ctrl_tech == "loso":
                            splitter = LeaveOneGroupOut()
                            assert splitter.get_n_splits(groups=data['site']) in [7,8]
                            test_idxs = [test_idx for _,test_idx in splitter.split(data["X"], groups=data['site'])]
                        else:
                            splitter = StratifiedKFold(n_splits=n_outer_cv, shuffle=True, random_state=0)
                            test_idxs = [test_idx for _,test_idx in splitter.split(data["X"], y=labels[y])] # dd: not performing stratify_by_conf='group' cuz stratification compromises the testset purity as the labels of the testset affects the data splitting and reduces variance in data                                  
                        for trial in range(n_outer_cv):
                            settings.extend([{"conf_ctrl_tech":conf_ctrl_tech, "confs": conf_names,
                                              "io":io, "model_pipegrid":ML_model, 
                                              "trial":trial, 
                                              "test_idx":test_idxs[trial]}]) 
            print(f"running {len(settings)} different settings of [confound_control, input-output, ML-model, out_cv_trial]")
            if debug: 
                for i, setting in enumerate(settings):
                    setting_to_print = copy(setting)
                    setting_to_print.pop('test_idx', None)
                    setting_to_print["model_pipegrid"] = setting_to_print["model_pipegrid"][0].steps[-1][0].replace("model_", "")    
                    print("({}) \t {}".format(i, setting_to_print))
            # runs the experiments with each parameter combination in parallel and save the results in run_y_i.csv
            parallel(delayed(
                        conf_corr_run)(
                                    h5_file=h5_file, **setting,
                                    label_name=y, 
                                    save_dir=save_dir, n_inner_cv=n_inner_cv, run_pbcc=run_pbcc,
                                    parallelize=parallelize, n_permutes_per_trial=n_permutes_per_trial,
                                    permute_only_xy=permute_only_xy, 
                                    save_models=save_models, debug=debug, random_state=random_state) 
                     for random_state, setting in enumerate(settings))

            # stitch together the csv results that were generated in parallel and save in a single csv file        
            df = pd.concat([pd.read_csv(csv) for csv in glob(save_dir+"/run_*.csv")], ignore_index=True)
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')] # drop unnamed columns            
            df = df.sort_values(["io","technique", "model", "trial"]) # sort
            df.to_csv(join(save_dir, "run.csv"), index=False)

            # delete the temp csv files generated in parallel
            os.system(f"rm {save_dir}/run_*.csv")                         
                    
            # calculate the chi-square statistics between confounds and label if requested
            if run_chi_square and conf_names:
                run = run_chi_sq(data, label_names, conf_names)
                run.to_csv(join(save_dir, "chi-square.csv"), index=False)                

            data.close()
            
            runtime=str(datetime.now()-start_time).split(".")[0]
            print("TOTAL RUNTIME: {} secs".format(runtime))

#########################################################################################################################
if __name__ == "__main__": 
    runMLpipelines(
            h5_files=H5_FILES, ML_models=ML_MODELS,
            n_outer_cv=N_OUTER_CV, n_inner_cv=N_INNER_CV,
            conf_ctrl_techs=CONF_CTRL_TECHS,  run_confs=RUN_CONFS,  exclude_in_run_confs= EXCLUDE_IN_RUN_CONFS,
            n_jobs=N_JOBS,  parallelize=PARALLELIZE, n_permutations=N_PERMUTATIONS,  permute_only_xy=PERMUTE_ONLY_XY, 
            save_models=SAVE_MODELS,  debug=DEBUG, run_chi_square=RUN_CHI_SQUARE,  run_pbcc=RUN_PBCC)
   