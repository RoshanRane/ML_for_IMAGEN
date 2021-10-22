# #################################################################################
# """ IMAGEN Posthoc Analysis Visualization """
# # Author: JiHoon Kim, <jihoon.kim@fu-berlin.de>, 18th October 2021
# #
# import math
# import seaborn as sns
# import matplotlib.pyplot as plt
# from scipy.stats import shapiro, levene, ttest_ind, bartlett
# from statannot import add_stat_annotation
# import warnings
# warnings.filterwarnings('ignore')
# sns.set_style("darkgrid")

#!/usr/bin/env python
# coding: utf-8

#################################################################################
import os
import numpy as np
import pandas as pd 
from glob import glob
import matplotlib.pyplot as plt
import shap
import h5py
import pickle
from joblib import load
import parmap

class SHAP_visualization:
    def __init__(self, DATA_DIR="/ritter/share/data/IMAGEN"):
        """ Set up path
        
        Parameters
        ----------
        DATA_DIR : string, optional
            Directory IMAGEN absolute path
        
        """
        # Set the directory path: IMAGEN
        self.DATA_DIR = DATA_DIR   
        
    def get_model(self, MODEL_DIR):
        """ Load the model
        
        Parameters
        ----------
        MODEL_DIR : string
            Directory saved Model path
        
        Returns
        -------
        self.MODELS : dictionary
            model configuration
        
        Examples
        --------
        >>> from plot_result_posthoc import *
        >>> MODEL = SHAP_visualization()
        >>> DICT = MODEL.get_model(
        ...     'MODEL_DIR')                  # MODELS
        
        Notes
        -----
        All the models weighted files are contained in same folder.
        
        """
        
        models_dir = sorted(glob(MODEL_DIR))[-1]
        models = {}
        model_names = list(set([f.split("_")[0] for f in os.listdir(models_dir) if f.split(".")[-1]=="model"]))
        for model_name in model_names:
            models.update({model_name: [load(f) for f in glob(models_dir+f"/{model_name}_*.model")]})
        self.MODELS = models
        self.MODEL_NAME = model_names
        return self.MODELS
        
    def get_train_data(self, H5_DIR, group=False):
        """ Load the train data
        
        Parameters
        ----------
        H5_DIR : string
            Directory saved File path
        group : boolean
            If True then generate the gorup_mask
            
        Returns
        -------
        self.tr_X : numpy.ndarray
            Data, hdf5 file
        self.tr_X_col_names : numpy.ndarray
            X features name list
        self.tr_Other : list
            at least contain y, numpy.ndarray or other Group mask
            
        Examples
        --------
        >>> from plot_result_posthoc import *
        >>> DATA = SHAP_visualization()
        >>> tr_X, tr_X_col_names, tr_Other = DATA.get_train_data(
        ...     'H5_DIR')                                  # DATA
        
        """
        data = h5py.File(H5_DIR, 'r')
        print(data.keys(), data.attrs.keys())
        X = data['X'][()]
        X_col_names = data.attrs['X_col_names']
        self.tr_X = X
        self.tr_X_col_names = X_col_names
        
        y = data[data.attrs['labels'][0]][()]
        if group == True:
            sex_mask = data['sex'].astype(bool)[()]
            class_mask = data['Binge'][()].astype(bool)
            self.tr_Other = [y, sex_mask, class_mask]
        else:
            self.tr_Other = [y]
        X.shape, len(X_col_names)
        return self.tr_X, self.tr_X_col_names, self.tr_Other

    def get_holdout_data(self, H5_DIR, group=False):
        """ Load the holdout data
        
        Parameters
        ----------
        H5_DIR : string
            Directory saved File path
        group : boolean
            If True then generate the gorup_mask
            
        Returns
        -------
        self.ho_X : numpy.ndarray
            Data, hdf5 file
        self.ho_X_col_names : numpy.ndarray
            X features name list
        self.ho_Other : list
            at least contain y, numpy.ndarray or other Group mask
            
        Examples
        --------
        >>> from plot_result_posthoc import *
        >>> DATA = SHAP_visualization()
        >>> ho_X, ho_X_col_names, ho_Other = DATA.get_train_data(
        ...     'H5_DIR')                                  # DATA
        
        """
        data = h5py.File(H5_DIR, 'r')
        print(data.keys(), data.attrs.keys())
        X = data['X'][()]
        X_col_names = data.attrs['X_col_names']
        self.ho_X = X
        self.ho_X_col_names = X_col_names
        
        y = data[data.attrs['labels'][0]][()]
        if group == True:
            sex_mask = data['sex'][()]
            class_mask = data['Binge'][()]
            self.ho_Other = [y, sex_mask, class_mask]
        else:
            self.ho_Other = [y]
        X.shape, len(X_col_names)
        return self.ho_X, self.ho_X_col_names, self.ho_Other
    
    def get_list(self, MODELS, X):
        """ Generate the SHAP input value list
        
        Parameters
        ----------
        MODELS : dictionary
            model configuration
        X : numpy.ndarray
            Data, hdf5 file
        
        Returns
        -------
        self.INPUT : list
            SHAP input combination list
        
        Examples
        --------
        >>> from plot_result_posthoc import *
        >>> DATA = SHAP_visualization()
        >>> INPUT = DATA.get_list(
        ...     'MODELS',                 # MODELS
        ...     'X')                      # X
        
        Notes
        -----
        Expected output below:
        INPUT = [
            [['SVM-RBF'], X, 0],
            [['SVM-RBF'], X, 1],
            [['SVM-RBF'], X, 2],
            [['SVM-RBF'], X, 3],
            ...
            [['GB'],      X, 5],
            [['GB'],      X, 6]
        ]
        
        """
        INPUT = []
        for model_name in MODELS:
            for i, model in enumerate(MODELS[model_name]):
                LIST = [model_name.upper(), X, i]
                INPUT.append(LIST)
        self.INPUT = INPUT
        return self.INPUT
    
    def to_SHAP(self, INPUT, save = True):
        """ Generate the SHAP value
        
        Parameters
        ----------
        INPUT: list
            SHAP INPUT: Model name, X, and N - trial number
        save : boolean
            Defualt save the shap_value
        
        Examples
        --------
        >>> from plot_result_posthoc import *
        >>> DATA = SHAP_visualization()
        >>> _ = DATA.to_SHAP(
        ...     'INPUT',                   # INPUT
        ...     save=True)                 # save        
        
        Notes
        -----
        explainers generate the SHAP value
        
        """
        MODEL = INPUT[0]
        X = INPUT[1]
        N = INPUT[2]
        # 100 instances for use as the background distribution
        X100 = shap.utils.sample(X, 100) 
        for model_name in self.MODELS:
            if (model_name.upper() not in MODEL):
#                 print(f"skipping model {model_name}")
                continue
#             print(f"generating SHAP values for model = {model_name} ..")
            for i, model in enumerate(self.MODELS[model_name]):
                if i!=N:
#                     print(f"Skipping model '{model_name}': {i}' as it is taking too long")
                    continue
                if i==N:
#                     print(f"generating SHAP values for model = {model_name}:{i} ..")
                    explainer = shap.Explainer(model.predict, X100, output_names=["Healthy","AUD-risk"])
                    shap_values = explainer(X)
                if save == True:
                    if not os.path.isdir("explainers"):
                        os.makedirs("explainers")
                    with open(f"explainers/{model_name+str(i)}_multi.sav", "wb") as f:
                        pickle.dump(shap_values, f)
        

    def plot_SHAP(MODEL, DATA, PLOT):
        # 1. choose the model (i = [0:6])
        # 2. subgroup: triaining and holdout
        # 3. plot: summary_plot bar, dot and summary_plot
        pass