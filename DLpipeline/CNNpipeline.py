# dependencies and plotting
import os, sys
from os.path import join
from contextlib import redirect_stdout
from glob import glob
import h5py
import matplotlib.pyplot as plt 
import numpy as np
import gc

# path
from pathlib import Path
from datetime import datetime
from joblib.externals.loky.backend.context import get_context
# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from nitorch.initialization import weights_init

# load functions from nitorch
sys.path.insert(1,"/ritter/roshan/workspace/nitorch/")
from nitorch.transforms import  ToTensor, SagittalTranslate, SagittalFlip, IntensityRescale 
from nitorch.callbacks import EarlyStopping, ModelCheckpoint
from nitorch.trainer import Trainer
from nitorch.metrics import binary_balanced_accuracy, sensitivity, specificity
from nitorch.utils import *
# from nitorch.utils import count_parameters
from nitorch.initialization import weights_init
from nitorch.data import *


class myDataset(Dataset):
    """Class for manipulating the IMAGEN Dataset. Inherits from the torch Dataset class.
    Parameters
    ----------
    X: Input data, i.e MRI images.
    y: Labels for the data.
    transfrom: Function for transforming the data into the appropriate format.
    mask: A mask that can be applied to the data with load_nifti.
    z_factor: A zoom factor that can be applied to the data with load_nifti.
    dtype: The desired data type of the data.      
    """
    def __init__(self, X, y, transform=None, mask=None, z_factor=None):
        self.X = X
        self.y = y
        self.transform = transform
        self.mask = mask
        self.z_factor = z_factor
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        image = self.X[idx]
        if self.transform: image = self.transform(image)
        # use soft label? # label = 0.98 if self.y[idx]>=0.5 else 0.02 
        label = torch.FloatTensor([(self.y[idx]>=0.5)])
            
        sample = {"image" : image, "label" : label}
        return sample
    
#################################################################################################################

def cnn_pipeline(
    model,
    gpu,
    X, y, val_idx, X_test, y_test, 
    batch_size=4, num_epochs=35,
    criterion = nn.BCEWithLogitsLoss, criterion_params = {},
    optimizer = optim.Adam, optimizer_params = {"lr":1e-4, "weight_decay":1e-4},
    weights_pretrained=None,
    augmentations = [],
    metrics=['binary_balanced_accuracy'], earlystop_patience=5,
    output_dir="results/", save_model=True,
    run_id=0, debug=False,
    **kwargs):      
    """
    Parameters
    ----------
    network: The custom pytorch DL network 
    data: a dict containing 'X' and 'y'
    val_idx: 
    callbacks: Callbacks used during training.
    metrics: The metrics used to evaluate the model during training.
    run_id: unique index to use when saving models, training logs etc
    
    Returns
    ----------
    model: trained model
    results: training results and log as a pandas series
    """
    
    # write all outputs to a separate .log file
    logfname = f'run{run_id}'
    with open(join(output_dir, logfname)+".log", "a") as logfile:
        with redirect_stdout(logfile):
            
            # initialize DL model on a GPU
            net = model().cuda(gpu)
            
            start_time = datetime.now()
            params_log = {
                "run_id":        run_id,
                "n_samples":     len(y),
                "m__name":       model.__name__,
                "m__n_params":   count_parameters(net),
                "m__batch_size": batch_size,
                "m__num_epochs": num_epochs,
                "m__criterion":  f"{criterion.__name__}({criterion_params})",
                "m__optimizer":  f"{optimizer.__name__}({optimizer_params})",
                "m__augmentations": [type(aug).__name__ for aug in augmentations],
                "m__weights_pretrained": weights_pretrained if weights_pretrained else "None",
                "m__earlystop_patience": earlystop_patience}
            
            print(f"Starting CNN_PIPELINE() with:\
            \n{params_log}\
            \ngpu: {gpu.__str__()}\
            \noutput_dir: {output_dir}\
            \n---------------- CNNpipeline starting ----------------")
            
            # prepare a dict to store all results
            result = {"model": str(net)} # save model full structure as str
            result.update(kwargs)
            result.update(params_log)
            
            # create a mask to distinguish between training & validation samples
            mask = np.ones(len(y), dtype=bool)
            mask[val_idx] = False
            
            # if provided, load a pretrained model
            if weights_pretrained:
                if os.path.exists(weights_pretrained.replace('*','{}',1).format(run_id)):
                    weights_pretrained = weights_pretrained.replace('*','{}',1).format(run_id)
                else:
                    weights_pretrained = np.random.choice(glob(weights_pretrained))
                print("Initializing model weights using a pretrained model:",weights_pretrained)
                net.load_state_dict(torch.load(weights_pretrained))
            else:
                net.apply(weights_init) # todo make this a function argument            
            
            # calculate the balancing loss weights, if requested (incase of unbalanced datasets)
            if ('pos_weight' in criterion_params) and criterion_params['pos_weight']=='balanced':
                # weight should be calculated as (n_neg_examples/n_pos_examples)
                _, c = np.unique(y[mask], return_counts=True)
                weight = c[0]/c[1]
                print("weighting possitive classes by {:.2f} in the loss function".format(weight))
                criterion_params['pos_weight'] = torch.FloatTensor([weight]).cuda(gpu)

            criterion = criterion(**criterion_params).cuda(gpu)
            optimizer = optimizer(net.parameters(), **optimizer_params)
            
            main_metric = metrics[0].__name__ if metrics else "binary_balanced_accuracy"
            # configure callbacks 
            callbacks = []
            if earlystop_patience:
                callbacks.extend([
                    EarlyStopping(earlystop_patience, window=2,
                                  ignore_before=earlystop_patience, 
                                  retain_metric="loss", mode='min')]) # dd: do early stopping on the loss
            if save_model:
                callbacks.extend([
                    ModelCheckpoint(path=output_dir, prepend=logfname,
                                    store_best=True, window=2,
                                    ignore_before=earlystop_patience, 
                                    retain_metric=main_metric)])  

            # prepare training and validation data as Pytorch DataLoader objects
            transform = transforms.Compose(augmentations + [ToTensor()])
            
            train_data = myDataset(X[mask], y[mask], transform=transform)
            train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=2, shuffle=True,
                                      multiprocessing_context=get_context('loky')) #https://github.com/pytorch/pytorch/issues/44687
            val_data = myDataset(X[~mask], y[~mask], transform=transform)
            val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, drop_last=False)
            
            trainer = Trainer(net,
                        criterion, optimizer,
                        metrics=metrics, callbacks=callbacks,
                        device=gpu, prediction_type="binary")
                    
            # train model
            net, report = trainer.train_model(
                                    train_loader, val_loader,
                                    num_epochs=num_epochs)                   
            
            # Save the training metrics in the results dict
            for key in ["train_metrics", "val_metrics"]:
                for metric, value in report[key].items():
                    result[key.replace('_metrics','_curve_')+metric]=value
                    
            # VALIDATE AGAIN on the val_set to get the predicted probabilities
            print("----------------------\nEvaluation on validation data again: \n----------------------")
            trainer.model = report['best_model']
            model_outputs, true_labels, val_report = trainer.evaluate_model(val_loader, 
                                                                         metrics=metrics,
                                                                         return_results=True)                       
            # save the probabilities in the results
            def sigmoid(m): return 1/(1+np.exp(-m))            
            result["val_preds"] = sigmoid(torch.cat(model_outputs).float().cpu().numpy().reshape(-1))
            # cross-verify that the order of val_data was not shuffled and is in the same order as val_ids
            true_labels = torch.cat(true_labels).float().cpu().numpy().reshape(-1)
            assert (result['val_lbls'] == true_labels).all(), \
"val_loader's data is not in the expected order"
            
            result.update({'val_'+k:v[0] for k,v in val_report.items()})
            
            # TEST on an independent holdout_set, if available
            print("----------------------\nEvaluation on holdout data: \n----------------------")
            if X_test is not None:         
                holdout_data = myDataset(X_test, y_test, transform=transform)
                holdout_loader = DataLoader(holdout_data, batch_size=batch_size, shuffle=False, drop_last=False)
                model_outputs, true_labels, hold_report = trainer.evaluate_model(holdout_loader, 
                                                         metrics=metrics, 
                                                         return_results=True)
                result["hold_preds"] = sigmoid(torch.cat(model_outputs).float().cpu().numpy().reshape(-1))
                true_labels = torch.cat(true_labels).float().cpu().numpy().reshape(-1)
                assert (result['hold_lbls'] == true_labels).all(), \
"holdout_loader's data is not in the expected order"
                
                result.update({'hold_'+k:v[0] for k,v in hold_report.items()})
                del holdout_loader
            
            # calculate total elapsed runtime
            runtime = datetime.now() - start_time
            result.update({"runtime":int(runtime.total_seconds())})
            # save result as a dataframe
            pd.DataFrame([result]).to_csv(join(output_dir, logfname+'.csv'))
            
            print("---------------- CNNpipeline completed ----------------")
            print("Ran for {}s and achieved val_score ({})= {:.2f}%".format(
                str(runtime).split(".")[0],
                main_metric, result["val_"+main_metric]*100))
            
            del net, optimizer, criterion, transform, trainer
            del train_loader, val_loader, report
            gc.collect()
            torch.cuda.empty_cache()
#     return best_model.cpu(), result

#################################################################################################################
    
def check_gpu_status(gpus):
    c = torch.cuda
    devices = [torch.device(f'cuda:{gpu}' if c.is_available() else 'cpu') for gpu in gpus]
    print(f"--------------------------------------------------------------------\
          \nTorch version:     {torch.__version__}\
          \nCUDA available?    {c.is_available()}\
          \ngraphic name:      {c.get_device_name()}\
          \nusing GPUs:        {[dev.__str__() for dev in devices]}\
          \nmemory alloc:      {[str(round(c.memory_allocated(dev)/1024**3,1))+' GB' for dev in devices]}\
          \nmemory cached:     {[str(round(c.memory_reserved(dev)/1024**3,1))+' GB' for dev in devices]}\
          \nrandom_seed:       {torch.initial_seed()}\
          \n--------------------------------------------------------------------\
          ")
    return devices

from scipy.stats import chi2_contingency, chi2
def run_chi_sq(data, confs):

    df = pd.DataFrame()
    for c in confs:
        chi, p, dof, _ = chi2_contingency(pd.crosstab(data['y'], data[c]))
        result = {"y":'y', "c":c, "chi":chi, "p-value":p, "dof":dof}
        df = df.append(result, ignore_index=True)
    return df