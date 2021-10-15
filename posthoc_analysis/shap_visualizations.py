#!/usr/bin/env python
# coding: utf-8


import pandas as pd 
from glob import glob
from os.path import join 
import os, sys
from scikits.bootstrap import ci
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")

sys.path.insert(1, '../MLpipelines/')
from plotResults import *


# ### 1. Visualization

# In[7]:


models_dir = sorted(glob("../MLpipelines/results/newlbls-fu3-espad-fu3-19a-binge-*/*/"))[-1]
models_dir

import shap


# In[22]:


# load all trained models
from joblib import load 

models = {}
model_names = list(set([f.split("_")[0] for f in os.listdir(models_dir) if f.split(".")[-1]=="model"]))
for model_name in model_names:
    models.update({model_name: [load(f) for f in glob(models_dir+f"/{model_name}_*.model")]})

models['SVM-rbf'][0]


# In[23]:


# load the training data
import h5py

h5_dir = "/ritter/share/data/IMAGEN/h5files/newholdout-fu3-espad-fu3-19a-binge-n102.h5"
data = h5py.File(h5_dir, 'r')
data.keys(), data.attrs.keys()


# In[24]:


X = data['X'][()]
y = data[data.attrs['labels'][0]][()]
X_col_names = data.attrs['X_col_names'][()]

X100 = shap.utils.sample(X, 100) # 100 instances for use as the background distribution

X.shape, y.shape, len(X_col_names)


# In[ ]:


# shap_values_stored = {}
for model_name in models:
    
    if ( model_name.upper() not in ["SVM-RBF", "GB"]):
        print("skipping model {}".format(model_name))
        continue
    
    print("generating SHAP values for model = {} ..".format(model_name))
    for i, model in enumerate(models[model_name]):
        if i>2:
            print("skipping model '{}: {}' as it is taking too long".format(model_name, i))
            continue
        print(model)
        explainer = shap.Explainer(model.predict, X100, output_names=["Healthy","AUD-risk"])
        shap_values = explainer(X)
        
#         shap_values_stored.update({model_name+str(i): shap_values})        
        
        # 1. summary bar plot of feature importance
        shap.summary_plot(shap_values, features=X, feature_names=X_col_names, plot_type="bar")
        plt.title(model_name+": "+str(i))
        plt.savefig("figures/{}_bar.pdf".format(model_name+str(i)), bbox_inches='tight')

        # 2. swarm plot showing shap values vs feature values ordered by feature importance
        shap.summary_plot(shap_values, features=X, feature_names=X_col_names, plot_type="dot")
        plt.title(model_name+": "+str(i))
        plt.savefig("figures/{}_swarm.pdf".format(model_name+str(i)), bbox_inches='tight')

        # 3. bar plot showing feature shap value differences between sex groups
        shap.group_difference_plot(shap_values.values, group_mask=data['sex'][()].astype(bool), 
                           feature_names=X_col_names, max_display=10)
        plt.title(model_name+": "+str(i))
        plt.savefig("figures/{}_bar-sexdiff.pdf".format(model_name+str(i)), bbox_inches='tight')
