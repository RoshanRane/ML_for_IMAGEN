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
              dropna = False, figsize = (6,4), title=None, ax=None,
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
    
    if(dropna): 
        filtered_df = df[cat].dropna()
        filtered_df = filtered_df[filtered_df != 'NaN']
    # if no nans in the df then don't do anything
    elif(df[cat].isnull().values.any()):
        filtered_df = df[cat]
    else:
        filtered_df = df[cat].fillna('no-data')
    
    # histogram and density plots
    if (plt_type == 'hist+density'):
        filtered_df.plot.hist(grid = True, alpha=0.7, density =True, bins=bins, ax=ax)
        filtered_df.plot.density(legend = True)
        ax.set_ylabel("density of test subjects")
        
    # only histogram plots
    elif (plt_type == 'hist'):
        filtered_df.plot.hist(grid = True, alpha=0.7, legend = True, bins=bins, ax=ax)
        ax.set_ylabel("number of test subjects")
        ax.set_xlim(0,int(bins))
        
    # pie-chart plots
    elif (plt_type == 'pie'):            
        val = dict(filtered_df.astype('str').value_counts())
        ax.pie(list(val.values()), labels = list(val.keys()), autopct='%1.1f%%', shadow=True, startangle=90) 
        ax.axis('equal')
        
    # bar plot
    elif (plt_type == 'bar'): 
        filtered_df.value_counts(dropna=dropna).sort_index().plot.bar(sort_columns=True, grid=True, rot=90, ax=ax,  width=0.9)
        if print_bar_count:
            x_texts = list(filtered_df.value_counts().sort_index())
            for i, x in enumerate(x_texts):
                ax.text( i , x+(x/100)+1 , str(x))
        ax.set_ylabel("number of test subjects")
        
    # barh plot
    elif (plt_type == 'barh'): 
        filtered_df.value_counts(dropna=dropna).sort_index().plot.barh(ax=ax, width=0.9)
        
        if print_bar_count:
            y_texts = list(filtered_df.value_counts().sort_index())
            for i, y in enumerate(y_texts):
                ax.text(y+(y/100), i, str(y))
        ax.set_xlabel("number of test subjects")
    
    else:
        raise ValueError("Invalid value for plt_type. It can be one of 'hist+density', 'hist', 'pie', 'bar', barh") 