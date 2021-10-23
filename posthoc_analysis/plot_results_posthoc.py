#################################################################################
#!/usr/bin/env python
# coding: utf-8
""" IMAGEN Posthoc Analysis Visualization """
# Author: JiHoon Kim, <jihoon.kim@fu-berlin.de>, 23th October 2021
#
import math
import pandas as pd
import seaborn as sns
from scipy.stats import shapiro, levene, ttest_ind, bartlett
from statannot import add_stat_annotation
import warnings

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
    
def ml_plot(train, test, col):
    fig, axes = plt.subplots(nrows=3, ncols=6, figsize=(3*len(col), 18))
    # 0,0. Training set
    ax0 = sns.countplot(data=train, x='Model',
                       hue='Model PN', hue_order=['TP & FP', 'TN & FN'],
                       ax=axes[0,0], palette='Set2')
    axes[0,0].set_title(f'Training Set (n = {len(train["Session"].tolist())//4}) by MODEL')
        
    axes[0,0].legend(title='Model Prediction',loc='lower center')
    
    for p in ax0.patches:
        ax0.annotate('{:}'.format(p.get_height()), (p.get_x()+0.05, p.get_height()))
        
    # 0,1. Model Prediction
    ax1 = sns.violinplot(data=train, x="Model", y=col,
                         inner="quartile", split=True,
                         hue='Model PN', hue_order=['TP & FP', 'TN & FN'],
                         ax = axes[0,1], palette="Set3")
        
    axes[0,1].set_title(f'{col}, Training Set')
    
    axes[0,1].legend(title='Model Prediction',loc='lower center')
        
    add_stat_annotation(ax1, data=train, x='Model', y=col, 
                        test='t-test_ind',
                        hue='Model PN', hue_order=['TP & FP', 'TN & FN'],
                        box_pairs=[(("GB","TP & FP"),("GB","TN & FN")),
                                   (("LR","TP & FP"),("LR","TN & FN")),
                                   (("SVM-lin","TP & FP"),("SVM-lin","TN & FN")),
                                   (("SVM-rbf","TP & FP"),("SVM-rbf","TN & FN"))],
                        loc='inside', verbose=2, line_height=0.1)
    
    # 0,2. Prediction TF
    ax2 = sns.violinplot(data=train, x="Model", y=col,
                         inner="quartile", split=True,
                         hue='Predict TF', hue_order=['TP & TN', 'FP & FN'],
                         ax = axes[0,2], palette="Set1")
            
    axes[0,2].set_title(f'{col}, Training Set')
    
    axes[0,2].legend(title='Prediction TF',loc='lower center')
        
    add_stat_annotation(ax2, data=train, x='Model', y=col, 
                        test='t-test_ind',
                        hue='Predict TF', hue_order=['TP & TN', 'FP & FN'],
                        box_pairs=[(("GB","TP & TN"),("GB","FP & FN")),
                                   (("LR","TP & TN"),("LR","FP & FN")),
                                   (("SVM-lin","TP & TN"),("SVM-lin","FP & FN")),
                                   (("SVM-rbf","TP & TN"),("SVM-rbf","FP & FN"))],
                        loc='inside', verbose=2, line_height=0.1)
        
    # 0,3. Holdout set
    ax3 = sns.countplot(data=test, x='Model',
                        hue='Model PN', hue_order=['TP & FP', 'TN & FN'],
                        ax=axes[0,3], palette='Set2')
    axes[0,3].set_title(f'Holdout Set (n = {len(test["Session"].tolist())//4}) by MODEL')
    
    axes[0,3].legend(title='Model Prediction',loc='lower center')
    
    for p in ax3.patches:
        ax3.annotate('{:}'.format(p.get_height()), (p.get_x()+0.05, p.get_height()))
        
    # 0,4. Model Prediction
    ax4 = sns.violinplot(data=test, x="Model", y=col,
                         inner="quartile", split=True,
                         hue='Model PN', hue_order=['TP & FP', 'TN & FN'],
                         ax = axes[0,4], palette="Set3")
        
    axes[0,4].set_title(f'{col}, Holdout Set')
    
    axes[0,4].legend(title='Model Prediction',loc='lower center')
        
    add_stat_annotation(ax4, data=test, x='Model', y=col, 
                        test='t-test_ind',
                        hue='Model PN', hue_order=['TP & FP', 'TN & FN'],
                        box_pairs=[(("GB","TP & FP"),("GB","TN & FN")),
                                   (("LR","TP & FP"),("LR","TN & FN")),
                                   (("SVM-lin","TP & FP"),("SVM-lin","TN & FN")),
                                   (("SVM-rbf","TP & FP"),("SVM-rbf","TN & FN"))],
                        loc='inside', verbose=2, line_height=0.1)
    
    # 0,5. Prediction TF
    ax5 = sns.violinplot(data=test, x="Model", y=col,
                         inner="quartile", split=True,
                         hue='Sex', hue_order=['Male', 'Female'],
                         ax = axes[0,5], palette="Set1")
        
    axes[0,5].set_title(f'{col}, Holdout Set')
    
    axes[0,5].legend(title='Prediction TF',loc='lower center')
        
    add_stat_annotation(ax5, data=test, x='Model', y=col, 
                        test='t-test_ind',
                        hue='Predict TF', hue_order=['TP & TN', 'FP & FN'],
                        box_pairs=[(("GB","TP & TN"),("GB","FP & FN")),
                                   (("LR","TP & TN"),("LR","FP & FN")),
                                   (("SVM-lin","TP & TN"),("SVM-lin","FP & FN")),
                                   (("SVM-rbf","TP & TN"),("SVM-rbf","FP & FN"))],
                        loc='inside', verbose=2, line_height=0.1)
    
    # 1,0. Training set
    ax0 = sns.countplot(data=train, x='Model',
                        hue='Prob', hue_order=['TP', 'FP', 'TN', 'FN'],
                        ax=axes[1,0], palette='Set2')
    axes[1,0].set_title(f'Training Set (n = {len(train["Session"].tolist())//4}) by MODEL')
        
    axes[1,0].legend(title='Model Prediction',loc='lower center')
    
    for p in ax0.patches:
        ax0.annotate('{:}'.format(p.get_height()), (p.get_x()+0.05, p.get_height()))
        
    # 1,1. Model Prediction P
    train_P = train[train['Model PN']=='TP & FP']
    ax1 = sns.violinplot(data=train_P, x="Model", y=col,
                         inner="quartile", split=True,
                         hue='Prob', hue_order=['TP', 'FP'],
                         ax = axes[1,1], palette="Set3")
        
    axes[1,1].set_title(f'{col}, Training Set')
    
    axes[1,1].legend(title='Prediction Positive',loc='lower center')
        
    add_stat_annotation(ax1, data=train_P, x='Model', y=col, 
                        test='t-test_ind',
                        hue='Prob', hue_order=['TP', 'FP'],
                        box_pairs=[(("GB","TP"),("GB","FP")),
                                   (("LR","TP"),("LR","FP")),
                                   (("SVM-lin","TP"),("SVM-lin","FP")),
                                   (("SVM-rbf","TP"),("SVM-rbf","FP"))],
                        loc='inside', verbose=2, line_height=0.1)
    
    # 1,2. Model Prediction N
    train_N = train[train['Model PN']=='TN & FN']
    ax2 = sns.violinplot(data=train_N, x="Model", y=col,
                         inner="quartile", split=True,
                         hue='Prob', hue_order=['TN', 'FN'],
                         ax = axes[1,2], palette="Set1")
            
    axes[1,2].set_title(f'{col}, Training Set')
    
    axes[1,2].legend(title='Prediction Negative',loc='lower center')
        
    add_stat_annotation(ax2, data=train_N, x='Model', y=col, 
                        test='t-test_ind',
                        hue='Prob', hue_order=['TN', 'FN'],
                        box_pairs=[(("GB","TN"),("GB","FN")),
                                   (("LR","TN"),("LR","FN")),
                                   (("SVM-lin","TN"),("SVM-lin","FN")),
                                   (("SVM-rbf","TN"),("SVM-rbf","FN"))],
                        loc='inside', verbose=2, line_height=0.1)
        
    # 1,3. Holdout set
    ax3 = sns.countplot(data=test, x='Model',
                        hue='Prob', hue_order=['TP', 'FP', 'TN', 'FN'],
                        ax=axes[1,3], palette='Set2')
    axes[1,3].set_title(f'Holdout Set (n = {len(test["Session"].tolist())//4}) by MODEL')
    
    axes[1,3].legend(title='Model Prediction',loc='lower center')
    
    for p in ax3.patches:
        ax3.annotate('{:}'.format(p.get_height()), (p.get_x()+0.05, p.get_height()))
        
    # 1,4. Model Prediction P
    test_P = test[test['Model PN']=='TP & FP']
    ax4 = sns.violinplot(data=test_P, x="Model", y=col,
                         inner="quartile", split=True,
                         hue='Prob', hue_order=['TP', 'FP'],
                         ax = axes[1,4], palette="Set3")
        
    axes[1,4].set_title(f'{col}, Holdout Set')
    
    axes[1,4].legend(title='Prediction Positive',loc='lower center')
        
    add_stat_annotation(ax4, data=test_P, x='Model', y=col, 
                        test='t-test_ind',
                        hue='Prob', hue_order=['TP', 'FP'],
                        box_pairs=[(("GB","TP"),("GB","FP")),
                                   (("LR","TP"),("LR","FP")),
                                   (("SVM-lin","TP"),("SVM-lin","FP")),
                                   (("SVM-rbf","TP"),("SVM-rbf","FP"))],
                        loc='inside', verbose=2, line_height=0.1)
    
    # 1,5. Model Prediction N
    test_N = test[test['Model PN']=='TN & FN']
    ax5 = sns.violinplot(data=test_N, x="Model", y=col,
                         inner="quartile", split=True,
                         hue='Prob', hue_order=['TN', 'FN'],
                         ax = axes[1,5], palette="Set1")
        
    axes[1,5].set_title(f'{col}, Holdout Set')
    
    axes[1,5].legend(title='Prediction Negative',loc='lower center')
        
    add_stat_annotation(ax5, data=test_N, x='Model', y=col, 
                        test='t-test_ind',
                        hue='Prob', hue_order=['TN', 'FN'],
                        box_pairs=[(("GB","TN"),("GB","FN")),
                                   (("LR","TN"),("LR","FN")),
                                   (("SVM-lin","TN"),("SVM-lin","FN")),
                                   (("SVM-rbf","TN"),("SVM-rbf","FN"))],
                        loc='inside', verbose=2, line_height=0.1)
    
    # 2,0. Training set
    ax0 = sns.countplot(data=train, x='Model',
                        hue='Prob', hue_order=['TP', 'FP', 'TN', 'FN'],
                        ax=axes[2,0], palette='Set2')
    axes[2,0].set_title(f'Training Set (n = {len(train["Session"].tolist())//4}) by MODEL')
        
    axes[2,0].legend(title='Model Prediction',loc='lower center')
    
    for p in ax0.patches:
        ax0.annotate('{:}'.format(p.get_height()), (p.get_x()+0.05, p.get_height()))
        
    # 2,1. Prediction T
    train_T = train[train['Predict TF']=='TP & TN']
    ax1 = sns.violinplot(data=train_T, x="Model", y=col,
                         inner="quartile", split=True,
                         hue='Prob', hue_order=['TP', 'TN'],
                         ax = axes[2,1], palette="Set3")
        
    axes[2,1].set_title(f'{col}, Training Set')
    
    axes[2,1].legend(title='Prediction True',loc='lower center')
        
    add_stat_annotation(ax1, data=train_T, x='Model', y=col, 
                        test='t-test_ind',
                        hue='Prob', hue_order=['TP', 'TN'],
                        box_pairs=[(("GB","TP"),("GB","TN")),
                                   (("LR","TP"),("LR","TN")),
                                   (("SVM-lin","TP"),("SVM-lin","TN")),
                                   (("SVM-rbf","TP"),("SVM-rbf","TN"))],
                        loc='inside', verbose=2, line_height=0.1)
    
    # 2,2. Prediction F
    train_F = train[train['Predict TF']=='FP & FN']
    ax2 = sns.violinplot(data=train_F, x="Model", y=col,
                         inner="quartile", split=True,
                         hue='Prob', hue_order=['FP', 'FN'],
                         ax = axes[2,2], palette="Set1")
            
    axes[2,2].set_title(f'{col}, Training Set')
    
    axes[2,2].legend(title='Prediction False',loc='lower center')
        
    add_stat_annotation(ax2, data=train_F, x='Model', y=col, 
                        test='t-test_ind',
                        hue='Prob', hue_order=['FP', 'FN'],
                        box_pairs=[(("GB","FP"),("GB","FN")),
                                   (("LR","FP"),("LR","FN")),
                                   (("SVM-lin","FP"),("SVM-lin","FN")),
                                   (("SVM-rbf","FP"),("SVM-rbf","FN"))],
                        loc='inside', verbose=2, line_height=0.1)
        
    # 2,3. Holdout set
    ax3 = sns.countplot(data=test, x='Model',
                        hue='Prob', hue_order=['TP', 'FP', 'TN', 'FN'],
                        ax=axes[2,3], palette='Set2')
    axes[2,3].set_title(f'Holdout Set (n = {len(test["Session"].tolist())//4}) by MODEL')
    
    axes[2,3].legend(title='Model Prediction',loc='lower center')
    
    for p in ax3.patches:
        ax3.annotate('{:}'.format(p.get_height()), (p.get_x()+0.05, p.get_height()))
        
    # 2,4. Prediction T
    test_T = test[test['Predict TF']=='TP & TN']
    ax4 = sns.violinplot(data=test_T, x="Model", y=col,
                         inner="quartile", split=True,
                         hue='Prob', hue_order=['TP', 'TN'],
                         ax = axes[2,4], palette="Set3")
        
    axes[2,4].set_title(f'{col}, Holdout Set')
    
    axes[2,4].legend(title='Prediction True',loc='lower center')
        
    add_stat_annotation(ax4, data=test_T, x='Model', y=col, 
                        test='t-test_ind',
                        hue='Prob', hue_order=['TP', 'TN'],
                        box_pairs=[(("GB","TP"),("GB","TN")),
                                   (("LR","TP"),("LR","TN")),
                                   (("SVM-lin","TP"),("SVM-lin","TN")),
                                   (("SVM-rbf","TP"),("SVM-rbf","TN"))],
                        loc='inside', verbose=2, line_height=0.1)
    
    # 2,5. Prediction F
    test_F = test[test['Predict TF']=='FP & FN']
    ax5 = sns.violinplot(data=test_F, x="Model", y=col,
                         inner="quartile", split=True,
                         hue='Prob', hue_order=['FP', 'FN'],
                         ax = axes[2,5], palette="Set1")
        
    axes[2,5].set_title(f'{col}, Holdout Set')
    
    axes[2,5].legend(title='Prediction False',loc='lower center')
        
    add_stat_annotation(ax5, data=test_F, x='Model', y=col, 
                        test='t-test_ind',
                        hue='Prob', hue_order=['FP', 'FN'],
                        box_pairs=[(("GB","FP"),("GB","FN")),
                                   (("LR","FP"),("LR","FN")),
                                   (("SVM-lin","FP"),("SVM-lin","FN")),
                                   (("SVM-rbf","FP"),("SVM-rbf","FN"))],
                        loc='inside', verbose=2, line_height=0.1)

def sc_plot(IN, data, col):
    fig, axes = plt.subplots(nrows=4, ncols=len(col)+1, figsize=(6*len(col), 7*4))
    # By class    
    ax = sns.countplot(data=data, x="Class", order=['HC', 'AAM'],
                       ax=axes[0,0], palette="Set1")
    
    axes[0,0].set_title(f'Session {data["Session"].values[0]}' +
                        f' (n = {len(data["Session"].tolist())}) by CLASS')

    for p in ax.patches:
        ax.annotate('{:}'.format(p.get_height()), (p.get_x()+0.35, p.get_height()+2))

    for i, roi in enumerate(col):
        ax2 = sns.violinplot(data=data, x="Class", order=['HC', 'AAM'],
                             y=roi, inner="quartile", split=True,
                             ax = axes[0,i+1], palette="Set1")
#         ax2.set(ylim=(0, None))
        
        axes[0,i+1].set_title(f'{IN} - {" ".join(roi.split(" ")[:-1])}' +
                              f' (n = {str(len(data[roi].dropna()))})')
        
        add_stat_annotation(ax2, data=data, x='Class', order=['HC', 'AAM'],
                            y=roi, test='t-test_ind',
                            box_pairs=[(("HC"), ("AAM"))],
                            loc='inside', verbose=2, line_height=0.06)

    # By sex and class
    ax = sns.countplot(data=data, x="Sex", order=['Male', 'Female'],
                       hue='Class',hue_order=['HC', 'AAM'],
                       ax=axes[1,0], palette="Set2")
    
    axes[1,0].set_title(f'Session {data["Session"].values[0]}' +
                        f' (n = {len(data["Session"].tolist())}) by CLASS|SEX')
    
    for p in ax.patches:
        ax.annotate('{:}'.format(p.get_height()), (p.get_x()+0.15, p.get_height()+2))
        
    for i, roi in enumerate(col):
        ax2 = sns.violinplot(data=data, x="Sex", order=['Male', 'Female'],
                             y=roi, hue='Class', hue_order=['HC', 'AAM'],
                             inner="quartile", split=True,
                             ax = axes[1,i+1], palette="Set2")
        
#         ax2.set(ylim=(0, None))
        
        axes[1,i+1].set_title(f'{IN} - {" ".join(roi.split(" ")[:-1])}' +
                              f' (n = {str(len(data[roi].dropna()))})')
        
        add_stat_annotation(ax2, data=data, x='Sex', order=['Male', 'Female'],
                            y=roi, test='t-test_ind',
                            hue='Class', hue_order = ['HC', 'AAM'],
                            box_pairs=[(("Male","HC"), ("Male","AAM")),
                                       (("Female","HC"), ("Female","AAM")),
                                       (("Male","HC"),("Female","HC")),
                                       (("Male","AAM"),("Female","AAM"))],
                            loc='inside', verbose=2, line_height=0.06)

    # By class and sex
    ax = sns.countplot(data=data, x="Class", order=['HC', 'AAM'],
                       hue='Sex',hue_order=['Male', 'Female'],
                       ax=axes[2,0])
    
    axes[2,0].set_title(f'Session {data["Session"].values[0]}' +
                        f' (n = {len(data["Session"].tolist())}) by SEX|CLASS')
    
    for p in ax.patches:
        ax.annotate('{:}'.format(p.get_height()), (p.get_x()+0.15, p.get_height()+2))
    
    for i, roi in enumerate(col):
        ax2 = sns.violinplot(data=data, x="Class", order=['HC', 'AAM'],
                             y=roi, hue='Sex', hue_order=['Male', 'Female'],
                             inner="quartile", split=True,
                             ax = axes[2,i+1])
        
#         ax2.set(ylim=(0, None))
        
        axes[2,i+1].set_title(f'{IN} - {" ".join(roi.split(" ")[:-1])}' +
                              f' (n = {str(len(data[roi].dropna()))})')
        
        add_stat_annotation(ax2, data=data, x='Class', order=['HC', 'AAM'],
                            y=roi, test='t-test_ind',
                            hue='Sex', hue_order = ['Male', 'Female'],
                            box_pairs=[(("HC","Male"), ("AAM","Male")),
                                       (("HC","Female"), ("AAM","Female")),
                                       (("HC","Male"),("HC","Female")),
                                       (("AAM","Male"),("AAM","Female"))],
                            loc='inside', verbose=2, line_height=0.06)
        
    # By sex
    ax = sns.countplot(data=data, x="Sex", order=['Male', 'Female'],
                       ax=axes[3,0], palette="Paired")

    axes[3,0].set_title(f'Session {data["Session"].values[0]}' +
                        f' (n = {len(data["Session"].tolist())}) by SEX')

    for p in ax.patches:
        ax.annotate('{:}'.format(p.get_height()), (p.get_x()+0.35, p.get_height()+2))

    for i, roi in enumerate(col):
        ax2 = sns.violinplot(data=data, x="Sex", order=['Male', 'Female'], y=roi,
                             inner="quartile", split=True,
                             ax = axes[3,i+1], palette="Paired")
        
#         ax2.set(ylim=(0, None))
        
        axes[3,i+1].set_title(f'{IN} - {" ".join(roi.split(" ")[:-1])}' +
                              f' (n = {str(len(data[roi].dropna()))})')
        
        add_stat_annotation(ax2, data=data, x='Sex', 
                            y=roi, test='t-test_ind',
                            box_pairs=[(("Male"), ("Female"))],
                            loc='inside', verbose=2, line_height=0.06)
        
    return [data.groupby(['Session','Class'])[col].mean(),
            data.groupby(['Session','Sex','Class'])[col].mean(),
            data.groupby(['Session','Class','Sex'])[col].mean(),
            data.groupby(['Session','Sex'])[col].mean()]