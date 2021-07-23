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
from copy import deepcopy

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

# Local imports
from MLpipeline import *
from confounds import *

# Define settings for the experiment 
DATA_DIR = "/ritter/share/data/IMAGEN"
# N_SITES = 8
## CV loops
N_OUTER_CV = 7 # number of folds in inner crossvalidation for test score estimation
N_INNER_CV = 5 # number of folds in inner crossvalidation used for hyperparameter tuning
## Optional runs
RUN_CONFS = False
RUN_CHI_SQUARE = True
RUN_PBCC = False # run the prediction-based confound correction technique by Dinga et al. 2020
SAVE_MODELS = True # save the final trained models for technique==CB and io== {X-y}
CONF_CTRLS = ["cb"] # choose from ["baseline", "cb", "cr", "cb+cr"]

## Permutation tests
# Total number of permutation tests to run. Set to 0 to not perform any permutations. 
N_PERMUTATIONS = 0
PERMUTE_ONLY_XY = True
N_JOBS = 10 # parallel jobs
PARALLELIZE = False # within each MLPipeline trial, do you want to parallelize the permutation test runs too?
# if set to true it will run 1 trial with no parallel jobs and enables debug msgs
DEBUG = False
    
if DEBUG:
    N_OUTER_CV = 2
    N_INNER_CV = 2
    if N_PERMUTATIONS > 2:
        N_PERMUTATIONS = 2
    N_JOBS = 1 
    PARALLELIZE = False
        
# The ML pipelines to run and their corresponding hyperparameter grids as tuples i.e. (pipeline, grid)
pipesgrids = [
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

# The experiments dict is a grid where you can select which technique and io you want to 
experiments = {
    "technique": CONF_CTRLS,
    "trial": range(N_OUTER_CV), 
    "pipesgrids": pipesgrids,
}

# Here you can select which HDF5 files you want to include in analysis. 
# Each entry should be (file_name, k_features).
files = [ 
### main
#  ('/ritter/share/data/IMAGEN/h5files/newlbls-bl-audit-fu3-audit-freq-audit-quick-n614.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/newlbls-bl-audit-fu3-audit-total-audit-n687.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/newlbls-bl-audit-gm-fine-cluster-audit-growth-n759.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/newlbls-bl-espad-fu3-19a-binge-n620.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/newlbls-bl-espad-fu3-29d-onset-15-n654.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/newlbls-bl-espad-fu3-8b-frequency-n868.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/newlbls-bl-audit-fu3-audit2-amount-n728.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/newlbls-bl-espad-gm-fine-cluster-binge-growth-n849.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/newlbls-bl-our-combo-cluster-combined-ours-n732.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/newlbls-bl-phenotype-phenotype-combined-seo-n740.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/newlbls-fu2-audit-fu3-audit-freq-audit-quick-n628.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/newlbls-fu2-audit-fu3-audit-total-audit-n705.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/newlbls-fu2-audit-gm-fine-cluster-audit-growth-n713.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/newlbls-fu2-espad-fu3-19a-binge-n634.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/newlbls-fu2-espad-fu3-29d-onset-15-n666.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/newlbls-fu2-espad-fu3-8b-frequency-n742.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/newlbls-fu2-audit-fu3-audit2-amount-n614.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/newlbls-fu2-espad-gm-fine-cluster-binge-growth-n720.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/newlbls-fu2-our-combo-cluster-combined-ours-n625.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/newlbls-fu2-phenotype-phenotype-combined-seo-n782.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/newlbls-fu3-audit-fu3-audit-freq-audit-quick-n623.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/newlbls-fu3-audit-fu3-audit-total-audit-n708.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/newlbls-fu3-audit-gm-fine-cluster-audit-growth-n589.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/newlbls-fu3-espad-fu3-19a-binge-n650.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/newlbls-fu3-espad-fu3-29d-onset-15-n697.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/newlbls-fu3-espad-fu3-8b-frequency-n762.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/newlbls-fu3-audit-fu3-audit2-amount-n630.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/newlbls-fu3-espad-gm-fine-cluster-binge-growth-n614.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/newlbls-fu3-our-combo-cluster-combined-ours-n631.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/newlbls-fu3-phenotype-phenotype-combined-seo-n589.h5', 'all'),
    
### ALL LABEL COMBOS
#  ('/ritter/share/data/IMAGEN/h5files/amt-combos-audit-bl-audit2-amt-14yr-a-0-1-n465.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/amt-combos-audit-fu1-audit2-amt-16yr-a-0-2-n471.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/amt-combos-audit-fu2-audit2-amt-19yr-a-0-2-n511.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/amt-combos-audit-fu3-audit2-amt-22yr-a-0-2-n585.h5', 'all'),
###  ('/ritter/share/data/IMAGEN/h5files/amt-combos-espad-bl-prev31-amt-14yr-0-1-n724.h5', 'all'), all estimators failed
#  ('/ritter/share/data/IMAGEN/h5files/amt-combos-espad-fu1-prev31-amt-16yr-1-3-n600.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/amt-combos-espad-fu2-prev31-amt-19yr-1-3-n579.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/amt-combos-espad-fu3-prev31-amt-22yr-2-4-n796.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/audit-combos-audit-bl-audit-freq-audit-c-14yr-1-3-n865.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/audit-combos-audit-bl-audit-freq-audit-c-14yr-2-4-n910.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/audit-combos-audit-bl-audit-symp-audit-d-14yr-0-2-n961.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/audit-combos-audit-bl-audit-total-audit-14yr-3-4-n961.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/audit-combos-audit-bl-audit-total-audit-14yr-3-5-n928.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/audit-combos-audit-fu1-audit-freq-audit-c-16yr-1-3-n738.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/audit-combos-audit-fu1-audit-freq-audit-c-16yr-2-4-n751.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/audit-combos-audit-fu1-audit-symp-audit-d-16yr-0-2-n856.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/audit-combos-audit-fu1-audit-total-audit-16yr-3-4-n856.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/audit-combos-audit-fu1-audit-total-audit-16yr-3-5-n774.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/audit-combos-audit-fu2-audit-freq-audit-c-19yr-2-4-n748.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/audit-combos-audit-fu2-audit-symp-audit-d-19yr-0-2-n638.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/audit-combos-audit-fu2-audit-total-audit-19yr-3-4-n855.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/audit-combos-audit-fu2-audit-total-audit-19yr-3-5-n741.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/audit-combos-audit-fu3-audit-freq-audit-c-22yr-2-4-n800.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/audit-combos-audit-fu3-audit-symp-audit-d-22yr-0-2-n729.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/audit-combos-audit-fu3-audit-total-audit-22yr-3-4-n962.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/audit-combos-audit-fu3-audit-total-audit-22yr-3-5-n858.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/binge-combos-espad-bl-19a-binge-14yr-1-3-n685.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/binge-combos-espad-bl-19a-binge-14yr-onset-0-2-n596.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/binge-combos-espad-bl-19a-binge-14yr-onset-0-3-n557.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/binge-combos-espad-bl-19a-binge-14yr-onset-n724.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/binge-combos-espad-fu1-19a-binge-16yr-1-4-n613.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/binge-combos-espad-fu1-19a-binge-16yr-2-4-n731.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/binge-combos-espad-fu1-19a-binge-16yr-onset-0-4-n438.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/binge-combos-espad-fu1-19a-binge-16yr-onset-n791.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/binge-combos-espad-fu2-19a-binge-19yr-1-4-n635.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/binge-combos-espad-fu2-19a-binge-19yr-2-5-n615.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/binge-combos-espad-fu2-19a-binge-19yr-3-6-n597.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/binge-combos-espad-fu2-19a-binge-19yr-onset-0-4-n529.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/binge-combos-espad-fu2-19a-binge-19yr-onset-n828.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/binge-combos-espad-fu2-19b-bingeyr-19yr-2-4-n598.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/binge-combos-espad-fu3-19a-binge-22yr-1-4-n790.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/binge-combos-espad-fu3-19a-binge-22yr-2-5-n738.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/binge-combos-espad-fu3-19a-binge-22yr-3-6-n650.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/binge-combos-espad-fu3-19a-binge-22yr-onset-n944.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/binge-combos-espad-fu3-19b-bingeyr-22yr-2-4-n784.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/binge-combos-espad-fu3-19b-bingeyr-22yr-2-5-n638.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/binge-combos-espad-fu3-19c-bingemnt-22yr-0-1-n775.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/binge-combos-espad-fu3-19c-bingemnt-22yr-1-2-n775.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/binge-combos-espad-fu3-29d-binge-onset-14-16-n697.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/binge-combos-espad-fu3-29d-binge-onset-16-18-n785.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/freq-combos-espad-bl-8b-freqyr-14yr-1-3-n574.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/freq-combos-espad-bl-8c-freq-14yr-0-1-n621.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/freq-combos-espad-fu1-8b-freqyr-16yr-3-5-n630.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/freq-combos-espad-fu1-8c-freq-16yr-1-3-n587.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/freq-combos-espad-fu2-8b-freqyr-19yr-4-6-n627.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/freq-combos-espad-fu2-8c-freq-19yr-1-3-n587.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/freq-combos-espad-fu3-8b-freqyr-22yr-4-6-n709.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/freq-combos-espad-fu3-8c-freq-22yr-2-4-n713.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/gm-combos-audit-gm-cluster-audit-gm-0-3-n490.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/gm-combos-audit-gm-cluster-audit-gm-1-2-n966.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/gm-combos-audit-gm-cluster-audit-gm-1-3-n840.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/gm-combos-audit-gm-fine-cluster-audit-gm-fine-1-5-n639.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/gm-combos-audit-gm-fine-cluster-audit-gm-fine-2-4-n894.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/gm-combos-audit-gm-fine-cluster-audit-gm-fine-int-n803.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/gm-combos-audit-gm-fine-cluster-audit-gm-fine-slp-n569.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/gm-combos-espad-gm-cluster-binge-gm-0-1-n767.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/gm-combos-espad-gm-cluster-binge-gm-0-2-n496.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/gm-combos-espad-gm-cluster-binge-gm-1-2-n767.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/gm-combos-espad-gm-fine-cluster-binge-gm-fine-2-5-no6-n573.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/gm-combos-espad-gm-fine-cluster-binge-gm-fine-2-6-n570.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/gm-combos-espad-gm-fine-cluster-binge-gm-fine-3-6-n641.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/gm-combos-espad-gm-fine-cluster-binge-gm-fine-int-n560.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/gm-combos-espad-gm-fine-cluster-binge-gm-fine-slp-ext-n539.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/gm-combos-espad-gm-fine-cluster-binge-gm-fine-slp-n697.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/combo-combos-combol0u1-binge-gm-slp-binge-lt-audit-q1-audit-q-n1035.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/rsfmri-fu2-binge-19yr-2-4-l2u4-n614.h5','all'),
#  ('/ritter/share/data/IMAGEN/h5files/rsfmri-fu2-binge-gm-slpl3u5678-n719.h5','all'),
#  ('/ritter/share/data/IMAGEN/h5files/rsfmri-fu2-binge-gml3u6-n674.h5', 'all')
    
    
### Modalities
#  ('/ritter/share/data/IMAGEN/h5files/newlblst1w-fu3-espad-fu3-19a-binge-n696.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/newlblsdti-fu3-espad-fu3-19a-binge-n653.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/newlblsthickness-fu3-espad-fu3-19a-binge-n696.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/newlblsvolume-fu3-espad-fu3-19a-binge-n696.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/newlblsarea-fu3-espad-fu3-19a-binge-n696.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/newlblscurv-fu3-espad-fu3-19a-binge-n696.h5', 'all'),

### Full data
#  ('/ritter/share/data/IMAGEN/h5files/fulldata-bl-audit-fu3-audit-freq-audit-quick-n713.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/fulldata-bl-audit-fu3-audit-total-audit-n789.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/fulldata-bl-audit-gm-fine-cluster-audit-growth-n848.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/fulldata-bl-espad-fu3-19a-binge-n722.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/fulldata-bl-espad-fu3-29d-onset-15-n768.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/fulldata-bl-espad-fu3-8c-frequency-n778.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/fulldata-bl-espad-fu3-prev31-amount-n713.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/fulldata-bl-espad-gm-fine-cluster-binge-growth-n889.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/fulldata-bl-our-combo-cluster-combined-ours-n909.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/fulldata-bl-phenotype-phenotype-combined-seo-n841.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/fulldata-fu2-audit-fu3-audit-freq-audit-quick-n727.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/fulldata-fu2-audit-fu3-audit-total-audit-n807.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/fulldata-fu2-audit-gm-fine-cluster-audit-growth-n802.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/fulldata-fu2-espad-fu3-19a-binge-n736.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/fulldata-fu2-espad-fu3-29d-onset-15-n780.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/fulldata-fu2-espad-fu3-8c-frequency-n804.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/fulldata-fu2-espad-fu3-prev31-amount-n734.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/fulldata-fu2-espad-gm-fine-cluster-binge-growth-n883.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/fulldata-fu2-our-combo-cluster-combined-ours-n861.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/fulldata-fu2-phenotype-phenotype-combined-seo-n883.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/fulldata-fu3-audit-fu3-audit-freq-audit-quick-n722.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/fulldata-fu3-audit-fu3-audit-total-audit-n810.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/fulldata-fu3-audit-gm-fine-cluster-audit-growth-n678.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/fulldata-fu3-espad-fu3-19a-binge-n752.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/fulldata-fu3-espad-fu3-29d-onset-15-n811.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/fulldata-fu3-espad-fu3-8c-frequency-n831.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/fulldata-fu3-espad-fu3-prev31-amount-n744.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/fulldata-fu3-espad-gm-fine-cluster-binge-growth-n750.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/fulldata-fu3-our-combo-cluster-combined-ours-n848.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/fulldata-fu3-phenotype-phenotype-combined-seo-n690.h5', 'all'),
]

def conf_corr_run(f, y, k_features, p, test_idxs,
                save_dir, confs, n_inner_cv, run_pbcc, run_confs,
                parallelize, n_permutes_per_trial, permute_only_xy, 
                save_models, debug, random_state=None):
    
    start_time_this_thread = datetime.now()
    technique = p["technique"].lower()
    i, o = p["io"]
    pipe, grid = deepcopy(p["pipesgrids"])
    trial = p["trial"]
    model_name = pipe.steps[-1][0].replace("model_", "")
    
    # Perform feature selection only when it is requested and input is 'X'. Also do not perform for GradientBoost
    if (k_features!='all' and (i=="X") and (model_name!="GB")):
        pipe.steps.insert(1, ("feature_selection", SelectKBest(k=int(k_features))))
    
    # skip confound-based analysis if requested
    if not run_confs and ((i in confs) or (o in confs)):
        return
    
    print("--------------------------------------")
    print("Starting a new pipeline with:\n", p)
    
    m = MLpipeline(parallelize, random_state=random_state, debug=debug)
    # load X, y and confounds
    m.load_data(f, y=y, confs=confs, group_confs=True)
    # randomly split data into training and test set
    m.train_test_split(test_idx=test_idxs[trial]) 
                                 
    ### <START> Special conditions for each confound correction technique
    conf_corr_params = {}  
    stratify_by_conf = None
    n_samples_cc = m.n_samples_tv
    # 1) CounterBalancing
    if technique in ["over-cb", "under-cb", "overunder-cb", "cb"]:
        # Counterbalance for both sex and site, which is "group"
        oversample = True
        if technique == "under-cb":
            oversample=False
        elif technique == "overunder-cb":
            oversample=None
        else:
            oversample=True
        cb = CounterBalance(oversample, random_state=random_state, debug=debug)
        pipe.steps.insert(-1, ("conf_corr_cb", cb))
        conf_corr_params.update({"conf_corr_cb__groups": m.confs["group"]})
        # when output is not the label 'y', still perform counterbalancing across the label 'y'
        if (o in confs): conf_corr_params.update({"conf_corr_cb__cb_by": m.y}) 
        # calculate info about how CB changes the training sample size
        n_samples_cc = len(cb._get_cb_sampled_idxs(groups=m.confs["group"], cb_by=m.y)) 
        
    # 2) Confound Regression
    elif (technique in ["cr"]) and (i == "X"):
        cr = ConfoundRegressorCategoricalX(debug=debug)
        pipe.steps.insert(-1, ("conf_corr_cr", cr))
        conf_corr_params.update({"conf_corr_cr__groups": m.confs["group"]})

    # 3) CB-CR: counterbalancing sex and confound regression for site
    elif (technique == "cb+cr"): # todo remove: only here for Seo et al results
        # use the first conf listed in CONFS for cb and second for cv
        cb_conf, cv_conf = confs 
        # The CR part for second confound ONLY
        # NOTE: do CR first in the pipeline to prevent the CB sampling for affecting the sample size
        if (i == "X"): 
            cr = ConfoundRegressorCategoricalX(debug=debug)
            pipe.steps.insert(-1, ("conf_corr_cr", cr))
            conf_corr_params.update({"conf_corr_cr__groups": m.confs[cv_conf]})

        # Counterbalance for first confound ONLY 
        cb = CounterBalance(oversample=False, random_state=random_state, debug=debug)    
        pipe.steps.insert(-1, ("conf_corr_cb", cb))
        conf_corr_params.update({"conf_corr_cb__groups": m.confs[cb_conf]}) 
        if (o in confs): conf_corr_params.update({"conf_corr_cb__cb_by": m.y}) 
        # calculate info about how CB would change the training sample size
        n_samples_cc = len(cb._get_cb_sampled_idxs(groups=m.confs[cb_conf], cb_by=m.y)) 
        
    # 4) Separate folds for each sex or each site or each sex-site group combined
    # use sklearn.model_selection.LeaveOneGroupOut
    # todo implement other confound-control methods
    ### <END> Special conditions for each confound correction technique

    if (i in confs): m.change_input_to_conf(i, onehot=True) # todo: onehot is hardcoded as confounds are rn categorical
    if (o in confs): m.change_output_to_conf(o)
    
    # run pbcc only for X-y
    if ((i in confs) or (o in confs)): 
        run_pbcc=False
    
    # run permutation for other than X-y experiments?
    if permute_only_xy and ((i in confs) or (o in confs)):
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
        "io" : "{}-{}".format(i,o),
        "technique" : technique,
        "model" : model_name,
        "trial" : trial,
        "n_samples":(m.n_samples_tv + m.n_samples_test),
        "n_samples_cc":(n_samples_cc + m.n_samples_test),
        "i" : i,
        "o" : o,
        "i_is_conf" : (i in confs),
        "o_is_conf" : (o in confs),
    }
    # Append results
    result.update(run)    
        
    
    runtime = int((datetime.now() - start_time_this_thread).total_seconds())
    result.update({"runtime":runtime})
    print("Finished after {}s with test_score = {:.2f}".format(
        str(datetime.now() - start_time_this_thread).split(".")[0], result['test_score']*100))    
    df = pd.DataFrame([result])
    df.to_csv(join(save_dir, f"run_{y}_{random_state}.csv" ))
    
    # save models only for X-y experiments with confound_control_technique == CounterBalance
    if save_models and (i not in confs) and (o not in confs) and (technique == "cb"):
        dump(m.estimator, join(save_dir, f"{model_name}_{trial}.model"))

#########################################################################################################################

def main():
    
    # The total number of permutations that are run per trial
    N_PERMUTES_PER_TRIAL = N_PERMUTATIONS//N_OUTER_CV
    
    with Parallel(n_jobs=N_JOBS) as parallel:
        
        for f, k_features in files:

            print("========================================")
            print("Running MLpipeline on file:\n", f)
            start_time = datetime.now()
            print("time: ", start_time)
            # Create the folder in which to save the results
            SAVE_DIR = "results/{}/{}".format(
                os.path.basename(f).replace(".h5",""),
                start_time.strftime("%Y%m%d-%H%M"))
            if not os.path.isdir(SAVE_DIR): os.makedirs(SAVE_DIR)
            
            # load the data.h5 file
            data = h5py.File(f, "r")
            # determine the input-output combinations to run from the h5 file
            label_names = data.attrs["labels"].tolist()
            conf_names = data.attrs["confs"].tolist()
            data_size = len(data["X"])
            labels = pd.DataFrame({lbl :np.array(data[lbl]) for lbl in label_names})
            
            # for each output run the pipeline
            for i, y in enumerate(label_names): # todo remove this multi-label design as it isn't useful
                # prepare the "io"
                io_combinations = [("X", y)]
                # Same analysis approach
                io_combinations.extend([(c , y) for c in conf_names])
                if i==0: 
                    # run {X-conf} combinations only the first time
                    io_combinations.extend([("X", c) for c in conf_names])            
                experiments["io"] = io_combinations
                
                exps = ParameterGrid(experiments)                
            
                # pre-generate the test indicies for the outer CV (that will be run in parallel)
                splitter = StratifiedKFold(n_splits=N_OUTER_CV, shuffle=True, random_state=0)
                test_idxs = [test_idx for _,test_idx in splitter.split(data["X"], y=labels[y])]
                # dd: not performing stratify_by_conf='group' cuz stratification compromises the testset purity as the labels of the testset affects the data splitting and reduces variance in data
#                 splitter = LeaveOneGroupOut()
#                 assert N_SITES == splitter.get_n_splits(groups=data['site'])
#                 test_idxs = [test_idx for _,test_idx in splitter.split(data["X"], groups=data['site'])]
                # runs the experiments with each parameter combination in parallel and save the results in run_y_i.csv
                parallel(delayed(
                            conf_corr_run)(
                                         f, y, k_features, exp, test_idxs, SAVE_DIR, conf_names, 
                                         N_INNER_CV, RUN_PBCC, RUN_CONFS,
                                         PARALLELIZE, N_PERMUTES_PER_TRIAL, PERMUTE_ONLY_XY,
                                         SAVE_MODELS, DEBUG, random_state) 
                         for random_state, exp in enumerate(exps))

            # stitch together the csv results that were generated in parallel and save in a single csv file        
            df = pd.concat([pd.read_csv(csv) for csv in glob(SAVE_DIR+"/run_*.csv")], ignore_index=True)
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')] # drop unnamed columns            
            df = df.sort_values(["io","technique", "model", "trial"]) # sort
            df.to_csv(join(SAVE_DIR, "run.csv"), index=False)

            # delete the temp csv files generated in parallel
            os.system(f"rm {SAVE_DIR}/run_*.csv")                         
                    
            # calculate the chi-square statistics between confounds and label if requested
            if RUN_CHI_SQUARE and conf_names:
                run = run_chi_sq(data, label_names, conf_names)
                run.to_csv(join(SAVE_DIR, "chi-square.csv"), index=False)                

            data.close()
            
            runtime=str(datetime.now()-start_time).split(".")[0]
            print("TOTAL RUNTIME: {} secs".format(runtime))

#########################################################################################################################
if __name__ == "__main__":
    main()
   