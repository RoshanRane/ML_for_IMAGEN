# dependencies and plotting
import os, sys
from os.path import join
from contextlib import redirect_stdout
from glob import glob
import h5py
import matplotlib.pyplot as plt 
import numpy as np

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
        # Convert labels to binary; sometimes labels are not exactly 0 or 1
        label = self.y[idx] >= 0.5 
        label = torch.FloatTensor([label])
        
        if self.transform:
            image = self.transform(image)
            
        sample = {"image" : image, "label" : label}
        return sample
    
#################################################################################################################

def cnn_pipeline(
    model,
    gpu,
    X, y, val_idx,
    batch_size=4, num_epochs=35,
    criterion = nn.BCEWithLogitsLoss, criterion_params = {},
    optimizer = optim.Adam, optimizer_params = {"lr":1e-4, "weight_decay":1e-4},
    weights_pretrained=None,
    augmentations = [],
    metrics=[binary_balanced_accuracy], earlystop_patience=5,
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
            
            start_time = datetime.now()
            params_log = {
                "run_id":     run_id,
                "model":      model.__name__,
                "n_samples":  len(y),
                "m__batch_size": batch_size,
                "m__num_epochs": num_epochs,
                "m__criterion": f"{criterion.__name__}({criterion_params})",
                "m__optimizer": f"{optimizer.__name__}({optimizer_params})",
                "m__augmentations": [type(aug).__name__ for aug in augmentations],
                "m__weights_pretrained": weights_pretrained if weights_pretrained else "None",
                "m__earlystop_patience": earlystop_patience}
            
            print(f"Starting CNN_PIPELINE() with:\
            \n{params_log}\
            \ngpu: {gpu.__str__()}\
            \noutput_dir: {output_dir}\
            \n---------------------------------------------------------------")
            
            # initialize DL model on a GPU
            net = model().cuda(gpu)
            print(f"Trainable model parameters: {count_parameters(net)}")
            # if provided, load a pretrained model
            if weights_pretrained:
                if os.path.exists(weights_pretrained.replace('*','{}',1).format(run_id)):
                    weights_pretrained = weights_pretrained.replace('*','{}',1).format(run_id)
                else:
                    weights_pretrained = np.random.choice(glob(weights_pretrained))
                print("Initializing model weights using a pretrained model:",weights_pretrained)
                net.load_state_dict(torch.load(weights_pretrained))
            else:
                net.apply(weights_init) # todo make this a parameter

            criterion = criterion(**criterion_params).cuda(gpu)
            optimizer = optimizer(net.parameters(), **optimizer_params)
            
            main_metric = metrics[0].__name__ # todo: bug what if metrics is empty list?
            # configure callbacks 
            callbacks = []
            if earlystop_patience:
                callbacks.extend([
                    EarlyStopping(earlystop_patience, window=2,
                                  ignore_before=earlystop_patience, 
                                  retain_metric="loss", mode='min')]) # dd: do early stopping on the loss instead of the metric
            if save_model:
                callbacks.extend([
                    ModelCheckpoint(path=output_dir, prepend=logfname,
                                    store_best=True, window=2,
                                    ignore_before=earlystop_patience, 
                                    retain_metric=main_metric)])  

            # prepare training and validation data as Pytorch DataLoader objects
            transform = transforms.Compose(augmentations + [ToTensor()])
            # create a mask to distinguish between training & validation samples
            mask = np.ones(len(y), dtype=bool)
            mask[val_idx] = False
            train_data = myDataset(X[mask], y[mask], transform=transform)
            train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=4, shuffle=True,
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
            
            # prepare the report to save the results in a run*.csv
            best_model = report.pop('best_model')
            for key in ["train_metrics", "val_metrics"]:
                for metric, value in report.pop(key).items():
                    report[key.replace('metrics','')+metric]=value
            report["val_best_"+main_metric]=report.pop("best_metric")
            
            # run validation again on the val_set to get the predicted probabilities
            model_outputs, true_labels = trainer.evaluate_model(val_loader, metrics=[], return_preds=True)
            # save the probabilities in the kwargs
            kwargs["val_preds"] = sigmoid(torch.cat(model_outputs).float().cpu().numpy().reshape(-1))
            # cross-verify that the val data order was not shuffled and is in the same order as val_ids
            true_labels = torch.cat(true_labels).float().cpu().numpy().reshape(-1)
            assert (kwargs['val_lbls'] == true_labels).all(), "val_loader's data order is not in the expected order"
            
            result = kwargs
            result.update(report)     
            result.update(params_log)
            result.update({"runtime":int((datetime.now() - start_time).total_seconds())})
            # save as a dataframe
            pd.DataFrame([result]).to_csv(join(output_dir, logfname+'.csv'))

            print("Finished after {}s with val_score ({}) = {:.2f}%".format(
                str(datetime.now() - start_time).split(".")[0],
                main_metric, result["val_best_"+main_metric]*100))
            
            torch.cuda.empty_cache()
        
    return best_model, result

########################################################################################


def evaluate_models(test_loader, model_dirs, network, gpu, output_dir):
    """Evaluates a group of models and prints the respective binary classification 
       accuracy, specificity, and sensitivity as well as the ROC curves.
       
    Parameters
    ----------
    test_loader: loaded test set using torch's DataLoader class
    model_dirs: list of dirs where the best performing models from each trial was stored
    network: network class used for classification
        
    Returns
    -------
    None
    """
    metrics = []
#             trainer.visualize_training(report, metrics, save_fig_path=join(output_dir,logfname))
#             trainer.evaluate_model(val_loader, write_to_dir=join(output_dir,logfname))
    for trial, model_dir in enumerate(model_dirs):
        print(f"trial {trial}")

        pred_score = []
        all_preds = []
        all_labels = []
        net = network().cuda(gpu)
        net.load_state_dict(torch.load(model_dir))
        
        # set net to evaluation mode so that batchnorm and dropout layers are in eval mode
        net.eval()
        
        with torch.no_grad():
            for sample in test_loader:
                img = sample["image"]
                label = sample["label"]

                img = img.to(torch.device(gpu))

                output = net.forward(img)
                pred_score.append(torch.sigmoid(output).cpu().numpy().item())
                # NOTE: As mentioned ealier, our last layer is linear, and so we need to send it
                #       through a sigmoid and then threshold the output to convert it to binary
                #       labels.
                pred = torch.sigmoid(output) >= 0.5
                all_preds.append(pred.cpu().numpy().item())
                all_labels.append(label.numpy().item())

        balanced_acc = binary_balanced_accuracy(all_labels, all_preds)
        specif = specificity(all_labels, all_preds)
        sensi = sensitivity(all_labels, all_preds)
        print(f"Balanced Accuracy: {balanced_acc}")
        print(f"Specificity: {specif}")
        print(f"Sensitivity: {sensi}\n")
        # NOTE: The roc_curve function from sklearn uses true labels and the pred_score. This
        #       means it needs the non thresholded output of the last layer.
        fpr, tpr, _ = roc_curve(all_labels, pred_score)
        roc_auc = auc(fpr, tpr)
        print_roc_and_auc(fpr, tpr, roc_auc)
        plt.savefig(output_dir+f"/trial{trial}_holdout_roc.png")
        # Here we set the net back to training mode before moving on to the next net to evaluate.
        net.train()
        metrics.append([balanced_acc, specif, sensi])
    metrics = np.array(metrics)
    print("######## Final results ########")
    print(f"Binary balanced accuracy mean: {np.mean(metrics[:, 0])*100:.2f} %")
    print(f"Specificity mean: {np.mean(metrics[:, 1])*100:.2f} %")
    print(f"Sensitivity mean: {np.mean(metrics[:, 2])*100:.2f} %")
    
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


def sigmoid(m):
    return 1/(1+np.exp(-m))