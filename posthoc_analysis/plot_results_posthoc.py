#################################################################################
#!/usr/bin/env python
# coding: utf-8
""" IMAGEN Posthoc Analysis Visualization """
# Author: JiHoon Kim, <jihoon.kim@fu-berlin.de>, 4th November 2021
#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import shapiro, levene, ttest_ind, bartlett
from statannot import add_stat_annotation
import warnings

def ml_plot(train, test, col):
    # model prediction plot
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
                         hue='Predict TF', hue_order=['TP & TN', 'FP & FN'],
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
    # sex and class plot
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

def violin_plot(DATA, ROI):
    # violin plot
    for col in ROI:
        sns.set(style="whitegrid", font_scale=1)
        fig, axes = plt.subplots(nrows=1, ncols=len(DATA), figsize = ((len(DATA)+1)**2, len(DATA)+1))
        fig.suptitle(f'{col}', fontsize=15)
        for i, (Key, DF) in enumerate(DATA):
            axes[i].set_title(f'{Key} = {str(len(DF[col].dropna()))}')
            sns.violinplot(x="Class", y=col, data = DF, order=['HC', 'AAM'],
                           inner="quartile", ax = axes[i], palette="Set2")
            add_stat_annotation(ax = axes[i], data=DF, x="Class", y=col,
                                box_pairs = [("HC","AAM")], order=["HC","AAM"],
                                test='t-test_ind', text_format='star', loc='inside')
            
def session_plot(DATA, ROI):
    # session plot
    for (S, DF) in DATA:
        columns = ROI
        sns.set(style="whitegrid", font_scale=1.5)
        fig, axes = plt.subplots(nrows=1, ncols=len(columns)+1,
                                 figsize=((len(columns)+1)**2, len(columns)+1))
        sns.countplot(x="Class", hue='Sex', order=['HC', 'AAM'], data = DF,
                      ax = axes[0], palette="Set2").set(title=S)
        
        for i, j in enumerate(columns):
            axes[i+1].set_title(columns[i])
            sns.violinplot(x="Class", y=j, data=DF, order=['HC', 'AAM'],
                           inner="quartile", ax = axes[i+1], palette="Set1")
            add_stat_annotation(ax = axes[i+1], data=DF, x="Class", y=j,
                                box_pairs = [("HC","AAM")], order=["HC","AAM"],
                                test='t-test_ind', text_format='star', loc='inside')
            
def plot_SHAP(MODEL, DATA, PLOT):
    # plot: summary_plot bar, dot and summary_plot
    pass

def SHAP_table(DF, viz = False):
    # DTI type
    DTI0 = [i for i in zip(DF['SVM rbf0 name'], DF['sorted SVM rbf0 mean'], DF['sorted SVM rbf0 std']) if 'DTI_' in i[0]]
    DTI1 = [i for i in zip(DF['SVM rbf1 name'], DF['sorted SVM rbf1 mean'], DF['sorted SVM rbf1 std']) if 'DTI_' in i[0]]
    DTI2 = [i for i in zip(DF['SVM rbf2 name'], DF['sorted SVM rbf2 mean'], DF['sorted SVM rbf2 std']) if 'DTI_' in i[0]]
    DTI3 = [i for i in zip(DF['SVM rbf3 name'], DF['sorted SVM rbf3 mean'], DF['sorted SVM rbf3 std']) if 'DTI_' in i[0]]
    DTI4 = [i for i in zip(DF['SVM rbf4 name'], DF['sorted SVM rbf4 mean'], DF['sorted SVM rbf4 std']) if 'DTI_' in i[0]]
    DTI5 = [i for i in zip(DF['SVM rbf5 name'], DF['sorted SVM rbf5 mean'], DF['sorted SVM rbf5 std']) if 'DTI_' in i[0]]
    DTI6 = [i for i in zip(DF['SVM rbf6 name'], DF['sorted SVM rbf6 mean'], DF['sorted SVM rbf6 std']) if 'DTI_' in i[0]]
    # T1w Subcortical type
    SUBCOR0 = [i for i in zip(DF['SVM rbf0 name'], DF['sorted SVM rbf0 mean'], DF['sorted SVM rbf0 std']) if 'T1w_subcor_' in i[0]]
    SUBCOR1 = [i for i in zip(DF['SVM rbf1 name'], DF['sorted SVM rbf1 mean'], DF['sorted SVM rbf1 std']) if 'T1w_subcor_' in i[0]]
    SUBCOR2 = [i for i in zip(DF['SVM rbf2 name'], DF['sorted SVM rbf2 mean'], DF['sorted SVM rbf2 std']) if 'T1w_subcor_' in i[0]]
    SUBCOR3 = [i for i in zip(DF['SVM rbf3 name'], DF['sorted SVM rbf3 mean'], DF['sorted SVM rbf3 std']) if 'T1w_subcor_' in i[0]]
    SUBCOR4 = [i for i in zip(DF['SVM rbf4 name'], DF['sorted SVM rbf4 mean'], DF['sorted SVM rbf4 std']) if 'T1w_subcor_' in i[0]]
    SUBCOR5 = [i for i in zip(DF['SVM rbf5 name'], DF['sorted SVM rbf5 mean'], DF['sorted SVM rbf5 std']) if 'T1w_subcor_' in i[0]]
    SUBCOR6 = [i for i in zip(DF['SVM rbf6 name'], DF['sorted SVM rbf6 mean'], DF['sorted SVM rbf6 std']) if 'T1w_subcor_' in i[0]]
    # T2w Subcortical type
    COR0 = [i for i in zip(DF['SVM rbf0 name'], DF['sorted SVM rbf0 mean'], DF['sorted SVM rbf0 std']) if 'T1w_cor_' in i[0]]
    COR1 = [i for i in zip(DF['SVM rbf1 name'], DF['sorted SVM rbf1 mean'], DF['sorted SVM rbf1 std']) if 'T1w_cor_' in i[0]]
    COR2 = [i for i in zip(DF['SVM rbf2 name'], DF['sorted SVM rbf2 mean'], DF['sorted SVM rbf2 std']) if 'T1w_cor_' in i[0]]
    COR3 = [i for i in zip(DF['SVM rbf3 name'], DF['sorted SVM rbf3 mean'], DF['sorted SVM rbf3 std']) if 'T1w_cor_' in i[0]]
    COR4 = [i for i in zip(DF['SVM rbf4 name'], DF['sorted SVM rbf4 mean'], DF['sorted SVM rbf4 std']) if 'T1w_cor_' in i[0]]
    COR5 = [i for i in zip(DF['SVM rbf5 name'], DF['sorted SVM rbf5 mean'], DF['sorted SVM rbf5 std']) if 'T1w_cor_' in i[0]]
    COR6 = [i for i in zip(DF['SVM rbf6 name'], DF['sorted SVM rbf6 mean'], DF['sorted SVM rbf6 std']) if 'T1w_cor_' in i[0]]
    # Common Features
    set_DTI = (set([i[0] for i in DTI0]) & set([i[0] for i in DTI1])  & set([i[0] for i in DTI2]) & set([i[0] for i in DTI3]) &
               set([i[0] for i in DTI4]) & set([i[0] for i in DTI5]) & set([i[0] for i in DTI6]))
    set_T1w_Sub = (set([i[0] for i in SUBCOR0]) & set([i[0] for i in SUBCOR1]) & set([i[0] for i in SUBCOR2]) & 
                   set([i[0] for i in SUBCOR3]) & set([i[0] for i in SUBCOR4]) & set([i[0] for i in SUBCOR5]) & set([i[0] for i in SUBCOR6]))
    set_T1w_Cor = (set([i[0] for i in COR0]) & set([i[0] for i in COR1]) & set([i[0] for i in COR2]) &
                   set([i[0] for i in COR3]) & set([i[0] for i in COR4]) & set([i[0] for i in COR5]) & set([i[0] for i in COR6]))
    # Generate the table
    d = [["DTI", len(DTI0), len(DTI1), len(DTI2), len(DTI3), len(DTI4), len(DTI5), len(DTI6), len(set_DTI)],
         ["T1w subcortical",len(SUBCOR0), len(SUBCOR1), len(SUBCOR2), len(SUBCOR3), len(SUBCOR4), len(SUBCOR5), len(SUBCOR6), len(set_T1w_Sub)],
         ["T1w cortical",len(COR0), len(COR1), len(COR2), len(COR3), len(COR4), len(COR5), len(COR6), len(set_T1w_Cor)]]
    df = pd.DataFrame(d, columns = ['Type','SVM-rbf 0','SVM-rbf1','SVM-rbf2','SVM-rbf3','SVM-rbf4','SVM-rbf5','SVM-rbf6','Intersection'])
    
    if viz == True:
#         print("DTI:    ",len(DTI0), len(DTI1), len(DTI2), len(DTI3), len(DTI4), len(DTI5), len(DTI6))
#         print("SUBCOR: ",len(SUBCOR0), len(SUBCOR1), len(SUBCOR2), len(SUBCOR3), len(SUBCOR4), len(SUBCOR5), len(SUBCOR6))
#         print("COR:    ",len(COR0), len(COR1), len(COR2), len(COR3), len(COR4), len(COR5), len(COR6))
        print(f"selected DTI (n={len(set_DTI)}): {set_DTI} \n\n"
              f"selected T1w Subcortical: (n={len(set_T1w_Sub)}): {set_T1w_Sub} \n\n"
              f"selected T1w Cortical: (n={len(set_T1w_Cor)}): {set_T1w_Cor} \n\n")
    return df
