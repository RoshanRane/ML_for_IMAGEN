import pandas as pd 
import numpy as np
from glob import glob
from os.path import join 
import os 
import matplotlib.pyplot as plt
import seaborn as sns
import re
import sklearn.metrics as metrics


def plot_result(df_full, x="test_score", conf_ctrl=[], input_type='',
                no_confs=False, sorty_by=None, join=False, beautify_io=True):
    
    input_types_cnt = 0
    if isinstance(df_full, (list, tuple)):
        if isinstance(input_type, (list, tuple)) and len(input_type)==len(df_full):
            df_full = [dfi.assign(i_type=i_type) for dfi, i_type in zip(df_full, input_type)]
            input_types_cnt = len(input_type)
        else:
            df_full = [dfi.assign(i_type=input_type) for dfi in df_full]
        df = pd.concat(df_full)
        is_multi = True
    else:
        df = df_full.copy()
        df = df.assign(i_type=input_type)
        is_multi = False
    # drop confound-related predictions if requested
    if "i_is_conf" in df.columns and no_confs: 
        df = df[~(df.i_is_conf) & ~(df.o_is_conf)]
        
    # calculate p_values if permutation tests were run
    if f"permuted_{x}" in df.columns:
        df = calc_p_values(df, x)
    
    # hack: for compatability between old MLpipeline and new DLpipeline
    if "technique" not in df and "conf_ctrl_tech" in df:
        df = df.rename(columns={"conf_ctrl_tech":"technique"})
    # if conf_ctrl is not configured then automatically determine it
    if not conf_ctrl: 
        conf_ctrl = df["technique"].unique()
    else:
        df = df[df["technique"].isin(conf_ctrl)]
        
    ios = df["io"].unique() 
    
    # if the x metric is not in the results, then try to compute this metric from sklearn.metrics
    if x not in df.columns:
        df = compute_metric(df, x)
    
    # make io labels latex formated
    if beautify_io:
        df.loc[:,"io"] = df.apply(remap_io, axis=1)
    df[x] = df[x].apply(lambda x: x*100)   
    
    # setup the figure properties
    sns.set(style='whitegrid', context='paper')
    fig, axes = plt.subplots(1, len(conf_ctrl), 
                             sharex=True, sharey=True, 
                             dpi=120, figsize=(4*len(conf_ctrl), 1+0.4*(len(ios)+input_types_cnt)))
    if not isinstance(axes, np.ndarray): axes = [axes]
    fig.gca().set_xlim([0,100])    
        
    y="io"     
    if is_multi:
        # if sample_size are different for different io (i.e. they are from different data subsets)
        # then print this info in y_label
        df = combine_io_sample_size(df)
        y="io_n" 
    y_order = get_y_order(df, y, sorty_by)
    for ((t, dfi), ax) in zip(df.groupby("technique"), axes):   
        assert (len(set(y_order) - set(dfi[y].unique())))==0 or ('baseline-cb' in df["technique"].unique())
        all_models = dfi["model"].unique()
        # plotting details
        palette = sns.color_palette()
        ci, dodge, scale, errwidth, capsize = 95, 0.4, 0.4, 0.9, 0.08  
        ML_MODELS = ["LR", "SVM-lin", "SVM-rbf", "GB"]
        hue_order= [m for m in ML_MODELS if m in all_models]
        # if none of the ML_MODELS are in this df then don't differentiate by model
        if len(hue_order)==0: hue_order=None
        
        ax = sns.pointplot(y=y, x=x, order=y_order,
                           hue="model", hue_order=hue_order,
                           join=join, data=dfi, ax=ax,
                           ci=ci, errwidth=errwidth, capsize=capsize,
                           dodge=dodge, scale=scale, palette=palette)# todo bugfix ci is sensible only if each trial score is statistically independant
        ax.legend_.remove()
        ax.set_title("{} (ave. n={:.0f})".format(t.upper(), dfi["n_samples"].mean()))
        ax.set_xlabel(x)
        ax.set_ylabel("")

        # Add significance stars if permutation scores are available
        if f"permuted_{x}" in dfi.columns:
            # collect p_values as a dict of form {mean_accuracy: p_value}
            p_dict = {g[x].mean(): g["p_value"].iloc[0] 
                      for i, g in dfi.groupby(["io", "model"])}
            # filter out the error bars from all lines 
            # and store the position of their right edges to print the p_values at 
            err_bars_pos = [(l.get_xdata()[1], l.get_ydata()[0])
                            for l in ax.lines if l.get_ydata()[0]==l.get_ydata()[1]] 
            # also collect the position of the mean 'accuracy' points
            points_pos = [tuple(p_pos)  for c in ax.collections for p_pos in c.get_offsets()]
            
            for i, pos in enumerate(err_bars_pos):
                
                p_pos = points_pos[i]
                # check that the y position of both the point and error bar are the same
                assert p_pos[1] == pos[1], f"y position of mean pt ({p_pos}) and error bar ({pos}) don't match"
                # choose the error bar with the same mean accuracy point
                p =  p_dict[p_pos[0]] 
                ast = return_asterisks(p)
                
                ax.annotate(ast, np.array(pos) + (0.02, 0.06), 
                            color=palette[i//len(df[y].unique())], fontsize=5)

        # draw the chance line in the legend
        chance = (len(dfi[y].unique()))*[50] #todo
        for z, ch in enumerate(chance): 
            ax.axvline(x=ch, ymin=z/len(chance), ymax=(z+1)/(len(chance)), 
                        label="chance", c='gray', ls='--', lw=1.5)

    # add legend: add models info and chance label
    handles, legends = ax.get_legend_handles_labels()
    leg1 = fig.legend(handles[len(chance):], legends[len(chance):], 
                      title="Models",
                      bbox_to_anchor=[1.13, 0.89], loc="upper right", fancybox=True, frameon=True)
    # only choose the first 'chance' legend
    leg2 = fig.legend([handles[0]], [legends[0]], loc="center right")
    fig.add_artist(leg1)

    fig.tight_layout()
    
    return fig
    

# Calculate p-value
def calc_p_values(df, x="test_score", viz=False):
    
    df["p_value"] = np.nan
    grp_order = ["io", "technique", "model"]
    if 'i_type' in df and len(df['i_type'].unique())>1:
        grp_order.insert(0, 'i_type')
    groups = df.dropna(subset=[x, f"permuted_{x}"]).groupby(grp_order)   
    n_models = len(df["model"].unique())
    
    if viz:
        sns.reset_orig()
        n_rows = len(groups)//n_models
#         fig, axes = plt.subplots(n_rows, n_models, 
#                                  sharex=True, sharey=False,
#                                  figsize=(20, n_models*n_rows))
        ## paper plot jugaad
        fig, axes = plt.subplots(1, 3, 
                                 sharex=True, sharey=True,
                                 figsize=(12, 4))
        axes = np.ravel(axes)
        plt.xlim([0,1])
        
    for i, (g, rows) in enumerate(groups):
        
        p_vals = [] 
        rows = rows.filter(like=x)
        if viz:
            permuted_scores = []    
            true_scores = rows[x]
            
        for _, r in rows.iterrows():
            p_scores = np.array(eval(r[f'permuted_{x}']))
            # calc p_value = (C + 1)/(n_permutations + 1) 
            # where C is permutations whose score >= true_score 
            true_score = r[x]
            C = np.sum(p_scores >= r[x])
            pi = (C+1)/(len(p_scores)+1)
            p_vals.extend([pi])
            
            if viz: permuted_scores.extend([*p_scores])        
#         if np.std(permuted_scores)>=1:
#             print("[WARN] the p-values for {} have high variance across each test-set (trial). \
# Simply averaging the p-values across trials in such a case is not recommended.".format(g))  

        df.loc[(df[grp_order]==g).all(axis=1), ["p_value", ""]] = np.mean(p_vals)  
        
        if viz:
            ax = axes[i]
#             ax.set_title("Model={}".format(g[-1]))
#             if i%n_models == 0:
#                 ax.set_ylabel("{} with {}".format(*g[-3:-1]))
            ax.hist(permuted_scores, bins='auto', alpha=0.8)
            for true_score in true_scores:
                ax.axvline(true_score, color='r')
            # draw chance lines 
            if x in ["test_score"]:
                # chance is 14% for site prediction and 50% for y and sex predictions
                chance = 0.5 if g[0][-1]!='c' else (1/7)
                ax.axvline(chance, color='grey', linestyle="--", lw=1.5)
                ax.set_xlim(0.,1.)
            ## paper plot jugaad
            ax.set_xticklabels([str(item) for item in range(0,120, 120//len(ax.get_xticklabels()))])
            inp = "X_{{{}}}".format(["14yr", "19yr", "22yr"][i])
            out = "y_{{{binge}}}"
            ax.set_title(r"${} \longmapsto {}$".format(inp,out))
#             if i==0: ax.set_ylabel("distribution / counts")
            if i==1: ax.set_xlabel("Balanced accuracy (%)")
            if i==0:
                from matplotlib.lines import Line2D
                custom_lines = [Line2D([0], [0], color="tab:blue", markerfacecolor="tab:blue", marker='o', markersize=5, lw=0),
                                Line2D([0], [0], color="tab:red", lw=1, linestyle="--")]
                ax.legend(custom_lines, ['permuted score', 'model score'], loc='lower left')
            
    sns.set(style='whitegrid')
    return df


def return_asterisks(pval):
    if pd.isna(pval):
        ast = ""
    elif pval <= 0.001:
        ast = '***'
    elif pval <= 0.01:
        ast = '**'
    elif pval <= 0.05:
        ast = '*'
    else:
        ast = 'n.s.'
    return ast

def remap_io(r):
    if r.i_is_conf:
        i = "c_{{{}}}".format(r.i.replace("_", "\_"))
    elif r.i == 'X' and ("i_type" in r): 
        i = "X_{{{}}}".format(r.i_type.replace("_", "\_"))
    elif r.i == 'X': 
        i = "X"
    else: # then its a label
        i = "y_{{{}}}".format(r.i.replace("_", "\_"))
        

    if r.o_is_conf:
        o = "c_{{{}}}".format(r.o.replace("_", "\_"))
    else: # then its a label
        o = "y_{{{}}}".format(r.o.replace("_", "\_"))
    
    new_io = r"${} \longmapsto {}$".format(i, o) 
    return new_io

def combine_io_sample_size(df):
    df["io_n"] = ''
    # only estimate sample_size for the first technique to be plotted
    dfi = df.query("technique == '{}'".format(df.technique.unique()[0]))
    for io, dfii in dfi.groupby("io"):
        # set the same sample_size in io across all techniques to avoid different y_labels in subplots
        df.loc[(df["io"]==io), "io_n"] = "{}\n(n={:.0f})".format(io, dfii["n_samples"].mean())
    return df


def get_y_order(df, y="io", sorty_by=None):     
    if sorty_by is not None:
        dfi = df.query("technique == '{}'".format(df.technique.unique()[0]))[[y, sorty_by]] 
        y_order = dfi.groupby(y).mean().sort_values(by=sorty_by, ascending=True).index.values        
        return y_order
    else:
        return df[y].sort_values().unique()


## METRICS ##
def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    return tn/(tn+fp)

def compute_metric(df, metric_name):
    
    metrics_map = {"recall_score":    metrics.recall_score,
                   "precision_score": metrics.precision_score,
                   "sensitivity":     metrics.recall_score,
                   "specificity":     specificity_score,
                   "accuracy_score":  metrics.accuracy_score,
                   "f1_score":        metrics.f1_score,
                  }
    metric=None
    for k in metrics_map.keys():
        if metric_name.lower() in k:
            metric = metrics_map[k] 
            break
    if metric is None:
        raise ValueError(
            "ERROR: Invalid 'x' metric requested. Allowed metric_names are {}".format(
                metrics_map.keys()))
    
    df['y_true'] = df["test_lbls"].apply(lambda lbls: np.array(eval(lbls), dtype=int))
    df['y_pred'] = df["test_probs"].apply(lambda probs: np.argmax(np.array(eval(probs)), axis=1))
    
    df[metric_name] = df.apply(lambda x: metric(y_true=x.y_true, y_pred=x.y_pred), axis=1)
    
    return df


def calc_p_val(scores, rand_scores, print_it=False):
    p_vals = [] 
    for ii, true_score in enumerate(true_scores):
        C = np.sum(rand_scores >= true_score)
        pi = (C+1)/(len(rand_scores)+1)
        if print_it: print(f"p-val for {ii} with {true_score:.2f} = {pi:.4f}")
        p_vals.extend([pi])
    return p_vals