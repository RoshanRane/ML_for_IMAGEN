#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 12:01:35 2018

@author: roshan
"""

import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns

def plotGraph(df, cat, plt_type='bar', 
              dropna=False, figsize=(6,4), title=None, ax=None,
              bins=10, print_bar_count=True): 
    '''
    Plots a graph of type defined in 'plt_type' on the data in the column given by 'cat'.
    Arg::
    plt_type: can be one of 'hist+density', 'hist', 'pie', 'bar', 'barh'  
    '''   
    sns.set(color_codes=False)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    #set title
    if(title):
        ax.set_title(title)
    else:
        description = cat.split(" Index")[0]
        ax.set_title(description)
    
    data = df[cat]
    # if nans in the df then replace
    if(dropna): 
        data = data.dropna()
        data = data[(data!='NaN') or (data!="missing")]
    else:
        if data.dtype.name=='category' and "missing" not in data.cat.categories: 
            data.cat.add_categories("missing")
        data = data.fillna('missing')
    
    # histogram and density plots
    if (plt_type == 'hist+density'):
        data.plot.hist(grid = True, alpha=0.7, density=True, normed=True, bins=bins, ax=ax)
        data.plot.density(legend=True)
        ax.set_ylabel("density of subjects")
        
    # only histogram plots
    elif (plt_type == 'hist'):
        data.plot.hist(grid=True, alpha=0.7, bins=bins, ax=ax)
        ax.set_ylabel("number of subjects")

         # pie-chart plots
    elif (plt_type == 'pie'):            
        val = dict(data.astype('str').value_counts())
        ax.pie(list(val.values()), labels = list(val.keys()), autopct='%1.1f%%', shadow=True, startangle=270) 
        ax.axis('equal')
        
    # bar plot
    elif (plt_type == 'bar'): 
        data.value_counts(dropna=dropna).sort_index().plot.bar(sort_columns=True, grid=True, rot=75, ax=ax,  width=0.9)
        if print_bar_count:
            x_texts = list(data.value_counts().sort_index())
            for i, x in enumerate(x_texts):
                ax.text( i , x+(x/100)+1 , str(x))
        ax.set_ylabel("number of subjects")
        
    # barh plot
    elif (plt_type == 'barh'): 
        data.value_counts(dropna=dropna).sort_index().plot.barh(ax=ax, width=0.9)
        
        if print_bar_count:
            y_texts = list(data.value_counts().sort_index())
            for i, y in enumerate(y_texts):
                ax.text(y+(y/100), i, str(y))
        ax.set_xlabel("number of subjects")
    
    else:
        raise ValueError("Invalid value for plt_type. It can be one of 'hist+density', 'hist', 'pie', 'bar', barh") 