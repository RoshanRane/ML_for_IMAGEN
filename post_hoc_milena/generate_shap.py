import math
import time
import parmap
import pickle
import multiprocessing
from imagen_post_hoc_helper import *
import matplotlib.pyplot as plt
import seaborn as sns

num_cores = multiprocessing.cpu_count()
print(f'Available CPU cores: {num_cores}')
num_cores = math.floor(num_cores/3)
print(f'Set CPU cores: {num_cores}')

DATA_DIR = "/ritter/share/data/IMAGEN"
posthoc = IMAGEN_posthoc()

# analysis 2
naming = "causal1"
h5_model = "/ritter/share/projects/Roshan_share_data/post_hoc_milena/results/newlbls-clean-bl-espad-fu3-19a-binge-causal-onset1-n565/*/"
holdout = "newholdout-clean-bl-espad-fu3-19a-binge-causal-onset1-n90.h5"

# main analysis
MODELS = posthoc.get_model(h5_model)

holdout_dir = holdout
# load the holdout data
ho_X, ho_X_col_names, ho_list = posthoc.get_holdout_data(holdout_dir, group=True)
print(f"Holdout dataset: {ho_X.shape}, {len(ho_X_col_names)}, "
      f"{ho_list[0].shape}, {ho_list[1].shape}")

# generate the SHAP input list of the holdout ONLY SVM-rbf
ho_INPUT = posthoc.get_list(MODELS, ho_X, "All")
# print(f"Number of training set: {len(tr_INPUT)}\n\n" # , One example: {tr_INPUT[0:1]}\n\n"
print(f"Number of holdout set: {len(ho_INPUT)}")#, {ho_INPUT}")

# Multi processing
INPUT = ho_INPUT
start_time = time.time()
_ = parmap.map(posthoc.get_SHAP, INPUT, naming, pm_pbar=True, pm_processes=num_cores)
print("--- %s seconds ---" % (time.time() - start_time))

# analysis 1
naming = "causal0"
h5_model = "/ritter/share/projects/Roshan_share_data/post_hoc_milena/results/newlbls-clean-bl-espad-fu3-19a-binge-causal-onset0-n477/*/"
holdout = "newholdout-clean-bl-espad-fu3-19a-binge-causal-onset0-n78.h5"

# main analysis
MODELS = posthoc.get_model(h5_model)

holdout_dir = holdout
# load the holdout data
ho_X, ho_X_col_names, ho_list = posthoc.get_holdout_data(holdout_dir, group=True)
print(f"Holdout dataset: {ho_X.shape}, {len(ho_X_col_names)}, "
      f"{ho_list[0].shape}, {ho_list[1].shape}")

# generate the SHAP input list of the holdout ONLY SVM-rbf
ho_INPUT = posthoc.get_list(MODELS, ho_X, "All")
# print(f"Number of training set: {len(tr_INPUT)}\n\n" # , One example: {tr_INPUT[0:1]}\n\n"
print(f"Number of holdout set: {len(ho_INPUT)}")#, {ho_INPUT}")

# Multi processing
INPUT = ho_INPUT
start_time = time.time()
_ = parmap.map(posthoc.get_SHAP, INPUT, naming, pm_pbar=True, pm_processes=num_cores)
print("--- %s seconds ---" % (time.time() - start_time))