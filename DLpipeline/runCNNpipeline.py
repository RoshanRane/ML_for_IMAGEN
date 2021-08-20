import os, sys, inspect
from glob import glob
import h5py
import matplotlib.pyplot as plt 
import numpy as np
import random

from joblib import Parallel, delayed
from pathlib import Path
from datetime import datetime
# sklearn
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# load functions from nitorch
sys.path.insert(1,"../../nitorch/")
import nitorch
from nitorch.transforms import SagittalTranslate, SagittalFlip, IntensityRescale 
from nitorch.trainer import Trainer
from nitorch.metrics import binary_balanced_accuracy, sensitivity, specificity
# from nitorch.utils import count_parameters
from nitorch.initialization import weights_init
from nitorch.data import *

from CNNpipeline import *
from models import *
##############################################################################
### CONFIG start #############################################################

# H5_FILES = [(train_data, holdout_data), ...]
H5_FILES = [("/ritter/share/data/IMAGEN/h5files/fullbrain-fu3-z2-bingel3u6-n*.h5",
            "/ritter/share/data/IMAGEN/h5files/fullbrain-fu3-hold-z2-bingel3u6-n*.h5")]

CONF_CTRL_TECHS = ["none"] # todo test 'loo-site', 'loo-sex' 
K_FOLD_CV = True
N_CV_TRIALS = 5

GPUS= [1,2,3,4,5]
MAX_GPUS_IN_PARALLEL= N_CV_TRIALS # maximum number of GPUs to use at once 

RUN_CONFS = False # todo test
RAND_STATE = None
RUN_CHI_SQUARE = False 

DEBUG = False

# The DL model to train and it's corresponding hyperparameters as tuples i.e. (pipeline, grid)
MODEL_SETTINGS = [
    {
    "model": SixtyFourNet, 
    "batch_size": 8, "num_epochs": 50, "earlystop_patience": 8,
    "criterion": nn.BCEWithLogitsLoss, "criterion_params": {'pos_weight':'balanced'},
    "optimizer": optim.Adam, "optimizer_params":{"lr": 1e-4, "weight_decay": 1e-4},
    "weights_pretrained": "results/pretrained_adni/run*_model-best.h5",
    "augmentations": [SagittalFlip(prob=0.5), SagittalTranslate(dist=(-2, 2)), IntensityRescale(masked=False)]
    },
    {
    "model": SixtyFourNet, 
    "batch_size": 8, "num_epochs": 50, "earlystop_patience": 8,
    "criterion": nn.BCEWithLogitsLoss, "criterion_params": {'pos_weight':'balanced'},
    "optimizer": optim.Adam, "optimizer_params": {"lr": 1e-4, "weight_decay": 1e-4},
    "weights_pretrained": None,
    "augmentations": [SagittalFlip(prob=0.5), SagittalTranslate(dist=(-2, 2)), IntensityRescale(masked=False)]
    },
]
### CONFIG ends  #############################################################
##############################################################################

if DEBUG:
    N_CV_TRIALS = 2
    MAX_GPUS_IN_PARALLEL = N_CV_TRIALS
    RAND_STATE = 42
    MODEL_SETTINGS = [MODEL_SETTINGS[0]]
    MODEL_SETTINGS[0]["num_epochs"]=5
    MODEL_SETTINGS[0]["batch_size"]=2
    MODEL_SETTINGS[0]["earlystop_patience"]=0
    
    
def main():
    
    if RAND_STATE:
        random.seed(RAND_STATE)
        np.random.seed(RAND_STATE)
        torch.manual_seed(RAND_STATE)
        
    with Parallel(n_jobs=MAX_GPUS_IN_PARALLEL) as parallel:  
        
        for h5_file, h5_hold_file in H5_FILES:

            start_time = datetime.now()
            # Create the folder in which to save the results
            if DEBUG: 
                os.system("rm -rf results/debug_run 2> /dev/null")
                SAVE_DIR = "results/debug_run/{}".format(
                start_time.strftime("%Y%m%d-%H%M"))
            else:
                SAVE_DIR = "results/{}/{}".format(
                    os.path.basename(h5_file).replace(".h5","").replace("*",""),
                    start_time.strftime("%Y%m%d-%H%M"))
            if not os.path.isdir(SAVE_DIR): os.makedirs(SAVE_DIR)
            if not len(os.listdir(SAVE_DIR))==0: 
                print(f"[WARN] output dir '{SAVE_DIR}' is not empty.. might overwrite..")

            print(f"========================================================\
            \nRunning CNNpipeline on: {h5_file}\
            \nStart time:             {start_time}\
            \nSaving results at:      {SAVE_DIR}")

            # print GPU status
            devices = check_gpu_status(GPUS)
        
            # load the data in h5_file
            with h5py.File(glob(h5_file)[0], 'r') as h5:
            # determine the input-output combinations to run from the h5 file
                conf_names = h5.attrs["confs"].tolist()            
                y = h5.attrs['labels'][0]
                data = {'X': np.array(h5['X']), 
                        'y': np.array(h5[y]),
                        'i': np.array(h5['i'])}
                for c in conf_names: data.update({c:np.array(h5[c])})
                    
                # also load holdout if it is provided
                data_hold = {'X':None, 'y':None, 'i':None}
                if h5_hold_file:
                    h5_hold = h5py.File(glob(h5_hold_file)[0], 'r')
                    data_hold = {'X': np.array(h5_hold['X']), 
                                 'y': np.array(h5_hold[y]),
                                 'i': np.array(h5_hold['i'])}
                    h5_hold.close()
                
            
            if DEBUG: 
                # only use the first 10 samples in debug mode
                rand_samples = np.random.randint(0, len(data['y']), size=10)
                for d in data: 
                    data[d] = data[d][rand_samples]
                
            # prepare the "io"
            io_combinations = [("X", y)]
            if RUN_CONFS:
                # skip confound-based analysis if not explicitly requested
                io_combinations.extend([(c , y) for c in conf_names]) # Same analysis approach
                io_combinations.extend([("X", c) for c in conf_names]) # Same analysis approach         

            # generate all setting combinations of 
            # (1) CONF_CTRL_TECHS  (2) INPUT_OUTPUT combination 
            # (3) MODEL pipeline   (4) N_CV_TRIALS trials so that they can be run on parallel GPUs
            settings = []
            for conf_ctrl_tech in CONF_CTRL_TECHS:
                for io in io_combinations:
                    for model_setting in MODEL_SETTINGS: 
                        # pre-generate the test indicies for the outer CV as they need to run in parallel
                        if "loo-" in conf_ctrl_tech:
                            conf_to_grp = conf_ctrl_tech.split("-")[-1]
                            splitter = LeaveOneGroupOut()
                            assert splitter.get_n_splits(groups=data[conf_to_grp]) in [7,8]
                            test_idxs = [test_idx for _,test_idx in splitter.split(data["X"], groups=data[conf_to_grp])]
                        elif K_FOLD_CV == True: 
                            splitter = StratifiedKFold(n_splits=N_CV_TRIALS, shuffle=True, random_state=RAND_STATE)
                            test_idxs = [test_idx for _,test_idx in splitter.split(data["X"], y=data['y'])] 
                        else: # random splits with overlapping test sets
                            test_idxs = [np.random.randint(0, len(data['y']), size=round(0.2*len(data['y']))
                                            ) for _ in range(N_CV_TRIALS)]
                        
                        for j, test_idx in enumerate(test_idxs):
                            settings.extend([{**model_setting,
                                              "conf_ctrl_tech":conf_ctrl_tech, 
                                              "i":io[0], "o":io[1], 
                                              "trial":j, 
                                              "val_idx": test_idx,                                              
                                              "val_ids": data['i'][test_idx], "val_lbls": data['y'][test_idx],
                                              "h5_train": h5_file, "h5_holdout": h5_hold_file,  
                                              "hold_ids": data_hold['i'], "hold_lbls": data_hold['y'],
                                              }]) 

            print(f"running {len(settings)} different combinations of [cross_val_folds, DL-models,\
 inputs-outputs, conf_ctrls] in parallel")
            
            # runs all the experiment combinations in parallel and save the results in `run_{i}.csv`
            parallel(
                delayed(
                        cnn_pipeline)(
                            X=data["X"], y=data["y"],
                            X_test=data_hold["X"], y_test=data_hold["y"],
                            **setting,
                            gpu=devices[i%len(devices)],
                            metrics=[binary_balanced_accuracy, sensitivity, specificity], 
                            save_model=True, output_dir=SAVE_DIR, 
                            run_id=i, debug=DEBUG) 
                     for i, setting in enumerate(settings))

            # stitch together the csv results that were generated in parallel and save in a single csv file        
            df = pd.concat([pd.read_csv(csv) for csv in glob(SAVE_DIR+"/run*.csv")], ignore_index=True)      
            # delete the temp csv files generated in parallel
            os.system(f"rm {SAVE_DIR}/run*.csv")  
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')] # drop unnamed columns            
            df = df.sort_values(['conf_ctrl_tech','i','o','run_id']) # sort
            df.to_csv(join(SAVE_DIR, "run.csv"), index=False)                       

            # calculate the chi-square statistics between confounds and label if requested
            if RUN_CHI_SQUARE and conf_names:
                run = run_chi_sq(data, conf_names)
                run.to_csv(join(SAVE_DIR, "chi-square.csv"), index=False)                
                
            runtime=str(datetime.now()-start_time).split(".")[0]
            print("TOTAL RUNTIME: {} secs".format(runtime))
            
            
##############################################################################################################

if __name__ == "__main__": main()