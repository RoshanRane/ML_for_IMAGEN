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
N_JOBS = 30 # parallel jobs
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
### New holdout set
#  ('/ritter/share/data/IMAGEN/h5files/newlbls-bl-audit-fu3-audit-freq-audit-quick-n614.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/newlbls-bl-audit-fu3-audit-total-audit-n687.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/newlbls-bl-audit-gm-fine-cluster-audit-growth-n759.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/newlbls-bl-espad-fu3-19a-binge-n620.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/newlbls-bl-espad-fu3-29d-onset-15-n654.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/newlbls-bl-espad-fu3-8c-frequency-n660.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/newlbls-bl-espad-fu3-prev31-amount-n620.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/newlbls-bl-espad-gm-fine-cluster-binge-growth-n781.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/newlbls-bl-our-combo-cluster-combined-ours-n793.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/newlbls-bl-phenotype-phenotype-combined-seo-n740.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/newlbls-fu2-audit-fu3-audit-freq-audit-quick-n628.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/newlbls-fu2-audit-fu3-audit-total-audit-n705.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/newlbls-fu2-audit-gm-fine-cluster-audit-growth-n713.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/newlbls-fu2-espad-fu3-19a-binge-n634.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/newlbls-fu2-espad-fu3-29d-onset-15-n666.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/newlbls-fu2-espad-fu3-8c-frequency-n686.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/newlbls-fu2-espad-fu3-prev31-amount-n641.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/newlbls-fu2-espad-gm-fine-cluster-binge-growth-n775.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/newlbls-fu2-our-combo-cluster-combined-ours-n745.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/newlbls-fu2-phenotype-phenotype-combined-seo-n782.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/newlbls-fu3-audit-fu3-audit-freq-audit-quick-n623.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/newlbls-fu3-audit-fu3-audit-total-audit-n708.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/newlbls-fu3-audit-gm-fine-cluster-audit-growth-n589.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/newlbls-fu3-espad-fu3-19a-binge-n650.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/newlbls-fu3-espad-fu3-29d-onset-15-n697.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/newlbls-fu3-espad-fu3-8c-frequency-n713.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/newlbls-fu3-espad-fu3-prev31-amount-n651.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/newlbls-fu3-espad-gm-fine-cluster-binge-growth-n642.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/newlbls-fu3-our-combo-cluster-combined-ours-n732.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/newlbls-fu3-phenotype-phenotype-combined-seo-n589.h5', 'all'),

### T1w only    
 ('/ritter/share/data/IMAGEN/h5files/newlblst1w-bl-audit-fu3-audit-freq-audit-quick-n769.h5', 'all'),
 ('/ritter/share/data/IMAGEN/h5files/newlblst1w-bl-audit-fu3-audit-total-audit-n863.h5', 'all'),
 ('/ritter/share/data/IMAGEN/h5files/newlblst1w-bl-audit-gm-fine-cluster-audit-growth-n962.h5', 'all'),
 ('/ritter/share/data/IMAGEN/h5files/newlblst1w-bl-espad-fu3-19a-binge-n783.h5', 'all'),
 ('/ritter/share/data/IMAGEN/h5files/newlblst1w-bl-espad-fu3-29d-onset-15-n834.h5', 'all'),
 ('/ritter/share/data/IMAGEN/h5files/newlblst1w-bl-espad-fu3-8c-frequency-n847.h5', 'all'),
 ('/ritter/share/data/IMAGEN/h5files/newlblst1w-bl-espad-fu3-prev31-amount-n784.h5', 'all'),
 ('/ritter/share/data/IMAGEN/h5files/newlblst1w-bl-espad-gm-fine-cluster-binge-growth-n994.h5', 'all'),
 ('/ritter/share/data/IMAGEN/h5files/newlblst1w-bl-our-combo-cluster-combined-ours-n1006.h5', 'all'),
 ('/ritter/share/data/IMAGEN/h5files/newlblst1w-bl-phenotype-phenotype-combined-seo-n901.h5', 'all'),
 ('/ritter/share/data/IMAGEN/h5files/newlblst1w-fu2-audit-fu3-audit-freq-audit-quick-n662.h5', 'all'),
 ('/ritter/share/data/IMAGEN/h5files/newlblst1w-fu2-audit-fu3-audit-total-audit-n750.h5', 'all'),
 ('/ritter/share/data/IMAGEN/h5files/newlblst1w-fu2-audit-gm-fine-cluster-audit-growth-n767.h5', 'all'),
 ('/ritter/share/data/IMAGEN/h5files/newlblst1w-fu2-espad-fu3-19a-binge-n681.h5', 'all'),
 ('/ritter/share/data/IMAGEN/h5files/newlblst1w-fu2-espad-fu3-29d-onset-15-n714.h5', 'all'),
 ('/ritter/share/data/IMAGEN/h5files/newlblst1w-fu2-espad-fu3-8c-frequency-n729.h5', 'all'),
 ('/ritter/share/data/IMAGEN/h5files/newlblst1w-fu2-espad-fu3-prev31-amount-n681.h5', 'all'),
 ('/ritter/share/data/IMAGEN/h5files/newlblst1w-fu2-espad-gm-fine-cluster-binge-growth-n828.h5', 'all'),
 ('/ritter/share/data/IMAGEN/h5files/newlblst1w-fu2-our-combo-cluster-combined-ours-n796.h5', 'all'),
 ('/ritter/share/data/IMAGEN/h5files/newlblst1w-fu2-phenotype-phenotype-combined-seo-n837.h5', 'all'),
 ('/ritter/share/data/IMAGEN/h5files/newlblst1w-fu3-audit-fu3-audit-freq-audit-quick-n674.h5', 'all'),
 ('/ritter/share/data/IMAGEN/h5files/newlblst1w-fu3-audit-fu3-audit-total-audit-n758.h5', 'all'),
 ('/ritter/share/data/IMAGEN/h5files/newlblst1w-fu3-audit-gm-fine-cluster-audit-growth-n629.h5', 'all'),
 ('/ritter/share/data/IMAGEN/h5files/newlblst1w-fu3-espad-fu3-19a-binge-n696.h5', 'all'),
 ('/ritter/share/data/IMAGEN/h5files/newlblst1w-fu3-espad-fu3-29d-onset-15-n746.h5', 'all'),
 ('/ritter/share/data/IMAGEN/h5files/newlblst1w-fu3-espad-fu3-8c-frequency-n757.h5', 'all'),
 ('/ritter/share/data/IMAGEN/h5files/newlblst1w-fu3-espad-fu3-prev31-amount-n697.h5', 'all'),
 ('/ritter/share/data/IMAGEN/h5files/newlblst1w-fu3-espad-gm-fine-cluster-binge-growth-n692.h5', 'all'),
 ('/ritter/share/data/IMAGEN/h5files/newlblst1w-fu3-our-combo-cluster-combined-ours-n785.h5', 'all'),
 ('/ritter/share/data/IMAGEN/h5files/newlblst1w-fu3-phenotype-phenotype-combined-seo-n629.h5', 'all'),

### DTI only
 ('/ritter/share/data/IMAGEN/h5files/newlblsdti-bl-audit-fu3-audit-freq-audit-quick-n620.h5', 'all'),
 ('/ritter/share/data/IMAGEN/h5files/newlblsdti-bl-audit-fu3-audit-total-audit-n696.h5', 'all'),
 ('/ritter/share/data/IMAGEN/h5files/newlblsdti-bl-audit-gm-fine-cluster-audit-growth-n768.h5', 'all'),
 ('/ritter/share/data/IMAGEN/h5files/newlblsdti-bl-espad-fu3-19a-binge-n627.h5', 'all'),
 ('/ritter/share/data/IMAGEN/h5files/newlblsdti-bl-espad-fu3-29d-onset-15-n659.h5', 'all'),
 ('/ritter/share/data/IMAGEN/h5files/newlblsdti-bl-espad-fu3-8c-frequency-n667.h5', 'all'),
 ('/ritter/share/data/IMAGEN/h5files/newlblsdti-bl-espad-fu3-prev31-amount-n626.h5', 'all'),
 ('/ritter/share/data/IMAGEN/h5files/newlblsdti-bl-espad-gm-fine-cluster-binge-growth-n787.h5', 'all'),
 ('/ritter/share/data/IMAGEN/h5files/newlblsdti-bl-our-combo-cluster-combined-ours-n802.h5', 'all'),
 ('/ritter/share/data/IMAGEN/h5files/newlblsdti-bl-phenotype-phenotype-combined-seo-n745.h5', 'all'),
 ('/ritter/share/data/IMAGEN/h5files/newlblsdti-fu2-audit-fu3-audit-freq-audit-quick-n628.h5', 'all'),
 ('/ritter/share/data/IMAGEN/h5files/newlblsdti-fu2-audit-fu3-audit-total-audit-n705.h5', 'all'),
 ('/ritter/share/data/IMAGEN/h5files/newlblsdti-fu2-audit-gm-fine-cluster-audit-growth-n713.h5', 'all'),
 ('/ritter/share/data/IMAGEN/h5files/newlblsdti-fu2-espad-fu3-19a-binge-n634.h5', 'all'),
 ('/ritter/share/data/IMAGEN/h5files/newlblsdti-fu2-espad-fu3-29d-onset-15-n666.h5', 'all'),
 ('/ritter/share/data/IMAGEN/h5files/newlblsdti-fu2-espad-fu3-8c-frequency-n686.h5', 'all'),
 ('/ritter/share/data/IMAGEN/h5files/newlblsdti-fu2-espad-fu3-prev31-amount-n641.h5', 'all'),
 ('/ritter/share/data/IMAGEN/h5files/newlblsdti-fu2-espad-gm-fine-cluster-binge-growth-n775.h5', 'all'),
 ('/ritter/share/data/IMAGEN/h5files/newlblsdti-fu2-our-combo-cluster-combined-ours-n745.h5', 'all'),
 ('/ritter/share/data/IMAGEN/h5files/newlblsdti-fu2-phenotype-phenotype-combined-seo-n782.h5', 'all'),
 ('/ritter/share/data/IMAGEN/h5files/newlblsdti-fu3-audit-fu3-audit-freq-audit-quick-n625.h5', 'all'),
 ('/ritter/share/data/IMAGEN/h5files/newlblsdti-fu3-audit-fu3-audit-total-audit-n711.h5', 'all'),
 ('/ritter/share/data/IMAGEN/h5files/newlblsdti-fu3-audit-gm-fine-cluster-audit-growth-n591.h5', 'all'),
 ('/ritter/share/data/IMAGEN/h5files/newlblsdti-fu3-espad-fu3-19a-binge-n653.h5', 'all'),
 ('/ritter/share/data/IMAGEN/h5files/newlblsdti-fu3-espad-fu3-29d-onset-15-n698.h5', 'all'),
 ('/ritter/share/data/IMAGEN/h5files/newlblsdti-fu3-espad-fu3-8c-frequency-n715.h5', 'all'),
 ('/ritter/share/data/IMAGEN/h5files/newlblsdti-fu3-espad-fu3-prev31-amount-n654.h5', 'all'),
 ('/ritter/share/data/IMAGEN/h5files/newlblsdti-fu3-espad-gm-fine-cluster-binge-growth-n643.h5', 'all'),
 ('/ritter/share/data/IMAGEN/h5files/newlblsdti-fu3-our-combo-cluster-combined-ours-n735.h5', 'all'),
 ('/ritter/share/data/IMAGEN/h5files/newlblsdti-fu3-phenotype-phenotype-combined-seo-n591.h5', 'all'),

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
### final 10 labels for all 3 TP
#  ('/ritter/share/data/IMAGEN/h5files/lbls-bl-audit-fu3-audit-freq-audit-quick-n613.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/lbls-bl-audit-fu3-audit-total-audit-n683.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/lbls-bl-audit-gm-fine-cluster-audit-growth-n726.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/lbls-bl-espad-fu3-19a-binge-n610.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/lbls-bl-espad-fu3-29d-onset-15-n660.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/lbls-bl-espad-fu3-8c-frequency-n660.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/lbls-bl-espad-fu3-prev31-amount-n613.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/lbls-bl-espad-gm-fine-cluster-binge-growth-n741.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/lbls-bl-our-combo-cluster-combined-ours-n706.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/lbls-bl-phenotype-phenotype-combined-seo-n709.h5', 'all'),
    
#  ('/ritter/share/data/IMAGEN/h5files/lbls-fu2-audit-fu3-audit-freq-audit-quick-n620.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/lbls-fu2-audit-fu3-audit-total-audit-n694.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/lbls-fu2-audit-gm-fine-cluster-audit-growth-n689.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/lbls-fu2-espad-fu3-19a-binge-n624.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/lbls-fu2-espad-fu3-29d-onset-15-n665.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/lbls-fu2-espad-fu3-8c-frequency-n682.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/lbls-fu2-espad-fu3-prev31-amount-n627.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/lbls-fu2-espad-gm-fine-cluster-binge-growth-n742.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/lbls-fu2-our-combo-cluster-combined-ours-n718.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/lbls-fu2-phenotype-phenotype-combined-seo-n746.h5', 'all'),
    
#  ('/ritter/share/data/IMAGEN/h5files/lbls-fu3-audit-fu3-audit-freq-audit-quick-n621.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/lbls-fu3-audit-fu3-audit-total-audit-n700.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/lbls-fu3-audit-gm-fine-cluster-audit-growth-n587.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/lbls-fu3-espad-fu3-19a-binge-n641.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/lbls-fu3-espad-fu3-29d-onset-15-n699.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/lbls-fu3-espad-fu3-8c-frequency-n714.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/lbls-fu3-espad-fu3-prev31-amount-n640.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/lbls-fu3-espad-gm-fine-cluster-binge-growth-n639.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/lbls-fu3-our-combo-cluster-combined-ours-n739.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/lbls-fu3-phenotype-phenotype-combined-seo-n593.h5', 'all'),

### DTI NEW
#  ('/ritter/share/data/IMAGEN/h5files/lblsnewdti-fu3-audit-fu3-audit-freq-audit-quick-n526.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/lblsnewdti-fu3-audit-fu3-audit-total-audit-n588.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/lblsnewdti-fu3-audit-gm-fine-cluster-audit-growth-n494.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/lblsnewdti-fu3-espad-fu3-19a-binge-n535.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/lblsnewdti-fu3-espad-fu3-29d-onset-15-n587.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/lblsnewdti-fu3-espad-fu3-8c-frequency-n606.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/lblsnewdti-fu3-espad-fu3-prev31-amount-n539.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/lblsnewdti-fu3-espad-gm-fine-cluster-binge-growth-n525.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/lblsnewdti-fu3-our-combo-cluster-combined-ours-n615.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/lblsnewdti-fu3-phenotype-phenotype-combined-seo-n497.h5', 'all'),
 
#  ('/ritter/share/data/IMAGEN/h5files/lblsnewdti-fu2-audit-fu3-audit-freq-audit-quick-n437.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/lblsnewdti-fu2-audit-fu3-audit-total-audit-n492.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/lblsnewdti-fu2-audit-gm-fine-cluster-audit-growth-n488.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/lblsnewdti-fu2-espad-fu3-19a-binge-n434.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/lblsnewdti-fu2-espad-fu3-29d-onset-15-n472.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/lblsnewdti-fu2-espad-fu3-8c-frequency-n486.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/lblsnewdti-fu2-espad-fu3-prev31-amount-n450.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/lblsnewdti-fu2-espad-gm-fine-cluster-binge-growth-n507.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/lblsnewdti-fu2-our-combo-cluster-combined-ours-n503.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/lblsnewdti-fu2-phenotype-phenotype-combined-seo-n517.h5', 'all'),
    
#  ('/ritter/share/data/IMAGEN/h5files/lblsnewdti-bl-audit-fu3-audit-freq-audit-quick-n514.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/lblsnewdti-bl-audit-fu3-audit-total-audit-n562.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/lblsnewdti-bl-audit-gm-fine-cluster-audit-growth-n596.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/lblsnewdti-bl-espad-fu3-19a-binge-n508.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/lblsnewdti-bl-espad-fu3-29d-onset-15-n551.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/lblsnewdti-bl-espad-fu3-8c-frequency-n560.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/lblsnewdti-bl-espad-fu3-prev31-amount-n506.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/lblsnewdti-bl-espad-gm-fine-cluster-binge-growth-n673.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/lblsnewdti-bl-our-combo-cluster-combined-ours-n579.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/lblsnewdti-bl-phenotype-phenotype-combined-seo-n596.h5', 'all'),
### ALL COMBOS
#  ('/ritter/share/data/IMAGEN/h5files/fsstats-area-fu3-binge-gml2u6-n668.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/fsstats-thickness-fu3-binge-gml2u6-n668.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/fsstats-thickness-volume-curv-fu3-binge-gml2u6-n668.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/fsstats-thickness-volume-fu3-binge-gml2u6-n668.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/fsstats-volume-fu3-binge-gml2u6-n668.h5', 'all'),
    
#  ('/ritter/share/data/IMAGEN/h5files/gm-combos-audit-gm-fine-cluster-audit-gm-fine-slp-n598.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/gm-combos-espad-gm-cluster-binge-gm-0-2-n628.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/gm-combos-espad-gm-fine-cluster-binge-gm-fine-2-6-n668.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/gm-combos-espad-gm-fine-cluster-binge-gm-fine-slp-n818.h5', 'all'),
    
#  ('/ritter/share/data/IMAGEN/h5files/combo-combos-combol0u1-binge-gm-slp-audit-gm-slp-binge-lt-binge-22yr-binge-gm-int-audit-q1-audit-q2-audit-gm-in-n829.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/combo-combos-combol0u1-binge-gm-slp-binge-lt-binge-22yr-audit-q1-audit-q-n829.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/combo-combos-combol0u2-binge-gm-slp-audit-gm-slp-binge-lt-binge-22yr-audit-q1-audit-q-n602.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/combo-combos-combol0u2-binge-gm-slp-binge-l-n610.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/combo-combos-combol0u2-binge-gm-slp-binge-lt-binge-22y-n668.h5', 'all'),
    
#  ('/ritter/share/data/IMAGEN/h5files/audit-combos-audit-bl-audit-freq-audit-c-14yr-1-3-n901.h5','all'),
#  ('/ritter/share/data/IMAGEN/h5files/audit-combos-audit-bl-audit-freq-audit-c-14yr-2-4-n943.h5','all'),
#  ('/ritter/share/data/IMAGEN/h5files/audit-combos-audit-bl-audit-symp-audit-d-14yr-0-2-n995.h5','all'),
#  ('/ritter/share/data/IMAGEN/h5files/audit-combos-audit-bl-audit-total-audit-14yr-3-4-n995.h5','all'),
#  ('/ritter/share/data/IMAGEN/h5files/audit-combos-audit-bl-audit-total-audit-14yr-3-5-n965.h5','all'),
#  ('/ritter/share/data/IMAGEN/h5files/audit-combos-audit-fu1-audit-freq-audit-c-16yr-1-3-n774.h5','all'),
#  ('/ritter/share/data/IMAGEN/h5files/audit-combos-audit-fu1-audit-freq-audit-c-16yr-2-4-n791.h5','all'),
#  ('/ritter/share/data/IMAGEN/h5files/audit-combos-audit-fu1-audit-symp-audit-d-16yr-0-2-n897.h5','all'),
#  ('/ritter/share/data/IMAGEN/h5files/audit-combos-audit-fu1-audit-total-audit-16yr-3-4-n897.h5','all'),
#  ('/ritter/share/data/IMAGEN/h5files/audit-combos-audit-fu1-audit-total-audit-16yr-3-5-n808.h5','all'),
#  ('/ritter/share/data/IMAGEN/h5files/audit-combos-audit-fu2-audit-freq-audit-c-19yr-2-4-n762.h5','all'),
#  ('/ritter/share/data/IMAGEN/h5files/audit-combos-audit-fu2-audit-symp-audit-d-19yr-0-2-n688.h5','all'),
#  ('/ritter/share/data/IMAGEN/h5files/audit-combos-audit-fu2-audit-total-audit-19yr-3-4-n898.h5','all'),
#  ('/ritter/share/data/IMAGEN/h5files/audit-combos-audit-fu2-audit-total-audit-19yr-3-5-n764.h5','all'),
#  ('/ritter/share/data/IMAGEN/h5files/audit-combos-audit-fu3-audit-freq-audit-c-22yr-2-4-n824.h5','all'),
#  ('/ritter/share/data/IMAGEN/h5files/audit-combos-audit-fu3-audit-symp-audit-d-22yr-0-2-n773.h5','all'),
#  ('/ritter/share/data/IMAGEN/h5files/audit-combos-audit-fu3-audit-total-audit-22yr-3-4-n1001.h5','all'),
#  ('/ritter/share/data/IMAGEN/h5files/audit-combos-audit-fu3-audit-total-audit-22yr-3-5-n894.h5','all'),
    
#  ('/ritter/share/data/IMAGEN/h5files/binge-combos-espad-bl-19a-binge-14yr-1-3-n715.h5','all'),
#  ('/ritter/share/data/IMAGEN/h5files/binge-combos-espad-bl-19a-binge-14yr-onset-0-2-n629.h5','all'),
#  ('/ritter/share/data/IMAGEN/h5files/binge-combos-espad-bl-19a-binge-14yr-onset-0-3-n594.h5','all'),
#  ('/ritter/share/data/IMAGEN/h5files/binge-combos-espad-bl-19a-binge-14yr-onset-n750.h5','all'),
#  ('/ritter/share/data/IMAGEN/h5files/binge-combos-espad-fu1-19a-binge-16yr-1-4-n650.h5','all'),
#  ('/ritter/share/data/IMAGEN/h5files/binge-combos-espad-fu1-19a-binge-16yr-2-4-n768.h5','all'),
#  ('/ritter/share/data/IMAGEN/h5files/binge-combos-espad-fu1-19a-binge-16yr-onset-0-4-n475.h5','all'),
#  ('/ritter/share/data/IMAGEN/h5files/binge-combos-espad-fu1-19a-binge-16yr-onset-n833.h5','all'),
#  ('/ritter/share/data/IMAGEN/h5files/binge-combos-espad-fu2-19a-binge-19yr-1-4-n664.h5','all'),
#  ('/ritter/share/data/IMAGEN/h5files/binge-combos-espad-fu2-19a-binge-19yr-2-5-n645.h5','all'),
#  ('/ritter/share/data/IMAGEN/h5files/binge-combos-espad-fu2-19a-binge-19yr-3-6-n626.h5','all'),
#  ('/ritter/share/data/IMAGEN/h5files/binge-combos-espad-fu2-19a-binge-19yr-onset-0-4-n543.h5','all'),
#  ('/ritter/share/data/IMAGEN/h5files/binge-combos-espad-fu2-19a-binge-19yr-onset-n868.h5','all'),
#  ('/ritter/share/data/IMAGEN/h5files/binge-combos-espad-fu2-19b-bingeyr-19yr-2-4-n625.h5','all'),
#  ('/ritter/share/data/IMAGEN/h5files/binge-combos-espad-fu3-19a-binge-22yr-1-4-n810.h5','all'),
#  ('/ritter/share/data/IMAGEN/h5files/binge-combos-espad-fu3-19a-binge-22yr-2-5-n761.h5','all'),
#  ('/ritter/share/data/IMAGEN/h5files/binge-combos-espad-fu3-19a-binge-22yr-3-6-n679.h5','all'),
#  ('/ritter/share/data/IMAGEN/h5files/binge-combos-espad-fu3-19a-binge-22yr-onset-n985.h5','all'),
#  ('/ritter/share/data/IMAGEN/h5files/binge-combos-espad-fu3-19b-bingeyr-22yr-2-4-n817.h5','all'),
#  ('/ritter/share/data/IMAGEN/h5files/binge-combos-espad-fu3-19b-bingeyr-22yr-2-5-n676.h5','all'),
#  ('/ritter/share/data/IMAGEN/h5files/binge-combos-espad-fu3-19c-bingemnt-22yr-0-1-n798.h5','all'),
#  ('/ritter/share/data/IMAGEN/h5files/binge-combos-espad-fu3-19c-bingemnt-22yr-1-2-n798.h5','all'),
#  ('/ritter/share/data/IMAGEN/h5files/binge-combos-espad-fu3-29d-binge-onset-14-16-n737.h5','all'),
#  ('/ritter/share/data/IMAGEN/h5files/binge-combos-espad-fu3-29d-binge-onset-16-18-n812.h5','all'),
    
#  ('/ritter/share/data/IMAGEN/h5files/faskdti-bl-binge-gml2u6-n826.h5','10000'),
#  ('/ritter/share/data/IMAGEN/h5files/faskdti-bl-binge-gml2u6-n826.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/faskdti-bl-z2-binge-gml2u6-n826.h5','1000'),
#  ('/ritter/share/data/IMAGEN/h5files/faskdti-bl-z2-binge-gml2u6-n826.h5','1000'),
#  ('/ritter/share/data/IMAGEN/h5files/faskdti-bl-z2-binge-gml2u6-n826.h5','all'),
#  ('/ritter/share/data/IMAGEN/h5files/faskdti-fu2-binge-gml2u6-n742.h5','10000'),
#  ('/ritter/share/data/IMAGEN/h5files/faskdti-fu2-binge-gml2u6-n742.h5', 'all'),
#  ('/ritter/share/data/IMAGEN/h5files/faskdti-fu2-z2-binge-gml2u6-n742.h5','1000'),
#  ('/ritter/share/data/IMAGEN/h5files/faskdti-fu2-z2-binge-gml2u6-n742.h5','all'),
#  ('/ritter/share/data/IMAGEN/h5files/fsstats-all-bl-seo-phenotypel0u2-n850.h5','all'),
#  ('/ritter/share/data/IMAGEN/h5files/fsstats-all-fu2-seo-phenotypel0u2-n786.h5','all'),
#  ('/ritter/share/data/IMAGEN/h5files/fsstats-all-fu3-seo-phenotypel0u2-n628.h5','all'),
    
#  ('/ritter/share/data/IMAGEN/h5files/rsfmri-fu2-binge-19yr-2-4-l2u4-n614.h5','all'),
#  ('/ritter/share/data/IMAGEN/h5files/rsfmri-fu2-binge-gm-slpl3u5678-n719.h5','all'),
#  ('/ritter/share/data/IMAGEN/h5files/rsfmri-fu2-binge-gml3u6-n674.h5', 'all')
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
   