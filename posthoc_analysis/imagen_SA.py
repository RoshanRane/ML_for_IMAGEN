import pandas as pd 
import numpy as np
from glob import glob
from os.path import join 
import os, sys
from scikits.bootstrap import ci
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from scipy import stats
from sklearn.metrics import *

# load all the labels used in the study as one pandas dataframe along with sex and site and subject ID information
import os, sys, inspect 
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from imagen_dataloader import *

def load_all_labels(x_tp="FU3", val_type='binarize', exclude_holdout=False): #, "FU2", "BL"
    lbl_combinations = [  
        ("ESPAD_FU3",        '19a',        3, 6,   'Binge'),
        ("ESPAD_GM_FINE",  'cluster',   2, 6,   'Binge_growth'),
        ("PHENOTYPE",      'Phenotype', 0, 2,   'Combined_seo'),
        ("OUR_COMBO",      'cluster', 0, 2,   'Combined_ours'),
        (f"AUDIT_FU3",     'audit_total', 4, 8,   'AUDIT'),
        (f"AUDIT_FU3",     'audit_freq', 3, 6,   'AUDIT_quick'),
        ("AUDIT_GM_FINE",  "cluster", [0,3], [2,5,6], 'AUDIT_growth'),
        ("ESPAD_FU3",      '29d', [16,17,18,19,20,21], [11,12,13,14],   'onset<15'),
        ("ESPAD_FU3", '8b', 4, 6, 'Frequency'),
        (f"AUDIT_FU3",   'audit2', 0, 2, 'Amount'),
    ]

    final_df = pd.DataFrame(index=Imagen(exclude_holdout=exclude_holdout).df_out.index)
    
    lbl_combos = lbl_combinations
    name = ""
    viz=0
    confs=['sex','site']

    for i, (csv, col, c0, c1, colname) in enumerate(lbl_combos):
        
        d = Imagen(exclude_holdout=exclude_holdout)

        if qs_is_raw_csv[csv]:
            dfq = pd.read_csv(qs[csv], usecols=["User code", col], dtype={"User code":str})
            dfq["ID"] = dfq["User code"].str.replace("-C", "").replace("-I", "").astype(int)
            dfq = dfq.drop("User code", axis=1)
        else:
            dfq = pd.read_csv(qs[csv], usecols=["ID", col])
        
        if val_type.lower() in ['binarize', 'categorize']:
            binarize, class0, class1=True, c0, c1
        else: 
            binarize, class0, class1=False, None, None
            
        d.load_label(dfq, col=col, viz=(viz>1), binarize=binarize, class0=class0, class1=class1, y_colname=colname)
        # drop the subjects in df_out that dont have imaging data
        # recreate the code of x.prepare_X() get the subject IDS to filter out the right subjects
        feature = pd.read_csv(join(d.DATA_DIR, f"IMAGEN_BIDS/sMRI-derivatives-clean_{x_tp}.csv"), index_col="ID")
        d.df_out = d.df_out.drop(labels= [idx for idx in d.df_out.index if idx not in feature.index])
        
        if val_type.lower() == 'categorize': # and val_type.lower() != 'categorize'
            d.df_out[d.all_labels[0]] = d.df_out[d.all_labels[0]].map({0:0, np.nan:1, 1:2})
    #         print(csv, len(feature.index))
        final_df[d.all_labels[0]] = d.df_out[d.all_labels[0]]

        # also get the confounds: add sex and site confounds
        if i==0:
            final_df[confs] = d.df[x_tp][confs]
            # remap sex values
            final_df['sex'] = final_df['sex'].map({'M':'Male','F':'Female'})
            final_df['site'] = final_df['site'].apply(lambda x: x.title())

        if viz: 
            plt.show()
            
    final_df = final_df.rename(columns={'onset<15':'Onset', 'sex':'Sex', 'site':'Site'})
#     final_df = final_df.dropna()
    
    return final_df



# plot the correlations between the different models
def get_corr(df, corr_type="r-squared", cols=None, rows=None):
    
    def r_squared(a, b): return (np.corrcoef(a, b)[0,1]**2)*100, np.NaN
    
    def accuracy(a, b): return accuracy_score(a, b)*100, np.NaN # no p_val calulated
    
    def chi_squared(a, b): return stats.chi2_contingency(pd.crosstab(a, b))[:2]
    
    # calculate the correlations
    corr_type = corr_type.lower()
    if corr_type in ['r', 'correlation', 'pearsonr', 'kendalltau', 'spearmanr']:
        vmin, vmax, cmap = -0.5, 0.5, 'RdBu'
        try:
            corr_type = getattr(stats, corr_type)
        except: # default to pearsonr correlation
            corr_type = stats.pearsonr
        fmt = '{:.2f}'
    elif corr_type in ['r2','r_squared','r-squared']:
        vmin, vmax, cmap = 0, 10, 'PuBu'
        corr_type = r_squared
        fmt = '{:.1f}%'
    elif corr_type in ['agreement', "accuracy"]:
        vmin, vmax, cmap = 0, 0.95, 'YlGnBu'
        corr_type = accuracy
        fmt = '{:.0f}%'
    elif corr_type in ['chi2', 'chi-squared']:
        vmin, vmax, cmap = 0, 100, 'YlGnBu'
        corr_type = chi_squared
        fmt = '{:.2f}'
#     elif 'jaccard' in corr_type:
#         vmin, vmax, cmap = 0.25, 1, 'YlGnBu'
#         corr_type = jaccard_score
#         fmt = '{:.2f}'
    else:
        print("error! unknown correlation type", corr_type)
    
    # if x and y not provided then perform for all combinations of row and cols
    if cols is None: cols = df.columns
    if rows is None: rows = df.columns    
        
    df_corr = pd.DataFrame(columns=cols, index=rows)
    df_p_vals = pd.DataFrame(columns=cols, index=rows).apply(pd.to_numeric)
    
    for rowi, row in df[rows].iteritems():
        for coli, col in df[cols].iteritems():
            # drop pairs with any NaNs
            nan_rows = pd.concat([row, col], axis=1).isna().any(axis=1)
            x, y = row[~nan_rows], col[~nan_rows]
            if nan_rows.sum()>=0.75*len(row): print('[WARN]{} vs {}: Only {} out of {} subjects used due to NaN values ({:.0f}% dropped)'.format(
                rowi, coli, len(row)-nan_rows.sum(), len(row), (nan_rows.sum()*100/len(row))))
            
            df_corr.loc[rowi, coli], df_p_vals.loc[rowi, coli] = corr_type(x, y)
            # convert p-vals to stars
#                 p_val = ''.join(['*' for t in [0.001,0.01,0.05] if p_val<=t])
#                 df_p_vals.loc[row, col] = p_val
            
    corr = df_corr.apply(pd.to_numeric).style.format(fmt).background_gradient(
                vmin=vmin, vmax=vmax, cmap=cmap).set_table_attributes("style='display:inline'").set_caption(corr_type.__name__.title())
    
    # also color style the p_vals
    def p_val_colors(v):
        cmap = plt.get_cmap('Reds')
        if v <= 0.001:
            return f"background-color: {matplotlib.colors.rgb2hex(cmap(0.8))};"  
        elif v <= 0.01:
            return f"background-color: {matplotlib.colors.rgb2hex(cmap(0.6))};"  
        elif v <= 0.05:
            return f"background-color: {matplotlib.colors.rgb2hex(cmap(0.5))};"  
        return None
    
    p_vals = df_p_vals.apply(pd.to_numeric).style.format('{:.4f}').applymap(
        p_val_colors).set_table_attributes("style='display:inline'").set_caption('P-vals')
        
    return corr, p_vals
        
### Store all predicted probablities from all models as one pandas dataframe each for each of the timepoints
def load_all_model_preds(df_all_labels, label='Binge'):
    all_model_predictions = {}
    for tp in ['fu3', 'fu2', 'bl']:

        df_model_results = df_all_labels[[label, 'Sex', 'Site']] 
        # add the 4 model predictions
        results = pd.read_csv(sorted(glob(f"../MLpipelines/results/newlbls-{tp}-espad-fu3-19a-binge-*/*/run.csv"))[-1])

        # drop conf-based experiments and non-conf-controlled experiments
        results = results[(results['technique']=='cb') & ~results['i_is_conf'] & ~results['o_is_conf']]
        # load the holdout
        holdout = pd.read_csv("../MLpipelines/results/holdout_all-tp_run.csv")
        holdout = holdout[(holdout['tp']==tp) & (holdout['technique']=='cb') & ~holdout['i_is_conf'] & ~holdout['o_is_conf']].drop(columns=["Unnamed: 0", "tp"])
        results = results.merge(holdout, how='outer', on=[c for c in holdout.columns if c in results.columns])

        for i, (model, dfi) in enumerate(results.groupby(["model"])):
            # get the test IDs from all 7 trials as one numpy array
            ids_all = np.array([id_lists for ids in dfi["test_ids"] for id_lists in eval(ids)])    
            # merge the holdout IDs
            # select the row (out of the 7 trials) with the best ROC_AUC score
            dfi_hold = holdout[holdout['model']==model].sort_values('holdout_roc_auc', ascending=False).iloc[0]
            ids_all_hold = np.array(eval(dfi_hold["holdout_ids"]))   
            ids_all = np.concatenate([ids_all, ids_all_hold])
        #     print(model, dfi_hold['holdout_score'], ids_all_hold)
            assert ids_all.shape[0] == np.unique(ids_all).shape[0] # just verify everything is fine

            # load the model prediction probabilities
            pred_probs_all = np.array([prob_lists for probs in dfi["test_probs"] for prob_lists in eval(probs)])[:,1]
            # get also holdout
            pred_probs_all_hold = np.array(eval(dfi_hold["holdout_preds"]))[:,1]
            pred_probs_all = np.concatenate([pred_probs_all, pred_probs_all_hold])   
            # binarize the probability to get predicted class
            y_pred_all = (pred_probs_all>0.5).astype(int)
            y_true_all = np.array([y_lists for y in dfi["test_lbls"] for y_lists in eval(y)])
            y_true_all_hold = np.array(eval(dfi_hold["holdout_lbls"]))
            y_true_all = np.concatenate([y_true_all, y_true_all_hold])   
            assert y_pred_all.shape[0] == y_true_all.shape[0] # just verify everything is fine
            data = pd.DataFrame(index=ids_all, data={model:y_pred_all, model+'_prob':pred_probs_all})
            df_model_results = df_model_results.join(data)
        # append the y_true values
        df_model_results[label] = pd.Series({ids:y_true for ids, y_true in zip(ids_all, y_true_all)})
    #     display(tp, df_model_results)
        all_model_predictions.update({tp:df_model_results})
        
    return all_model_predictions

######################################    PLOTTING    #########################################################
from matplotlib.lines import Line2D

def make_legend_lines(values, cmap):
    return [Line2D([], [], color=cmap(v), 
                    marker='|', ls='None', 
                    markersize=20, markeredgewidth=3) for v in values]

def plot_subject_classes(dfx, ax, confs=[], sort_order=[], lw=10, title='', fs=16, cmap=plt.cm.hot_r):
    
    df = dfx.copy()
    if sort_order: df= df.sort_values(sort_order) 
        
    legs = []
    # generate for confounds first
    cmap_conf=plt.cm.tab10
    leg_loc = ['lower right', 'center right']
    for ii, lbl in enumerate(confs):
        # convert the values to categorical values|
        indices = df[lbl].astype("category").cat.codes
        # Visualize the results
        ax.scatter(range(len(indices)), [ii+.5]*len(indices),
                   c=[cmap_conf(i) for i in indices],
                   marker='_', lw=lw)
        leg = ax.legend(make_legend_lines(indices.unique(), cmap=cmap_conf), 
                    df[lbl].unique(),
                    fontsize=fs, title=lbl.title(),
                    bbox_to_anchor=(1.19+0.055*ii, 0.4*ii), loc=leg_loc[ii]
                       )
        plt.setp(leg.get_title(),fontsize=fs+2)
        legs.extend([leg])
        
    # drop the confounds now
    df = df[[c for c in df.columns if c not in confs]]  
    
    # generate for labels (all the remaining columns)
    if sort_order:
        df = df.reindex(reversed(sort_order), axis=1)
    
    for ii, lbl in enumerate(df):
        # if categorical value convert to continuous for cmap
        vals = np.unique(df[lbl])
        if vals[-1]>1:
            cmap_val = np.arange(vals[0], vals[-1]+1)/(vals[-1]+0.01)+0.2
#             print(lbl, vals[-1], cmap_val)
            indices = df[lbl].apply(lambda x: cmap_val[int(x)]).values
        else:
            indices = df[lbl]
        # Visualize the results
        ax.scatter(range(len(indices)), [ii+len(confs)+.5]*len(indices),
                   c=[cmap(i) for i in indices],
                   marker='_', lw=lw, cmap=cmap, edgecolors='grey',
                   label=lbl)
        
    ax.set_yticks(np.arange(len(confs)+len(df.columns)) + .5)
    ax.set_yticklabels(confs+df.columns.values.tolist(), fontsize=fs)
    ax.set_xticklabels([])
    ax.set_ylabel('AAM phenotypes', fontsize=fs+1)
    ax.set_xlabel('Subjects', fontsize=fs+1)
    ax.set_xlim([0, len(df)])
    
    if np.unique(df[lbl])[-1]>1:
        leg3 = plt.legend(
                make_legend_lines(cmap_val[::-1], cmap=cmap), 
                ['Risky','Moderate','Safe'],
                fontsize=fs, 
                title="Alcohol user \ncategory",
                bbox_to_anchor=(1.22, 1), 
                loc="upper right")
        plt.setp(leg3.get_title(),fontsize=fs+2)
        # add the other conf legends
        [ax.add_artist(leg) for leg in legs]
    else:
        pass
    # todo
    
    plt.title(title, fontsize=fs+2)
    
    return ax


def plot_subject_classes_modelwise(model_results, use_probs, lbl='Binge', models=["SVM-rbf","GB","SVM-lin","LR"]):
    # prepare a df for plot_subject_classes() function
    for tp, df_model_results in model_results.items():
        print("=======================================\n              TP = ", tp.upper())
        # drop the probabilities for now
        df_ml =  df_model_results[[lbl, 'Sex', 'Site']]

        for m in models:
            if use_probs:
                df_ml = df_ml.assign(**{m: df_model_results[m+'_prob']}) #np.nan:1,
            else:
                df_ml = df_ml.assign(**{m: df_model_results[m].map({0:0, 1:2}, na_action='ignore')}) #np.nan:1,

        df_ml = df_ml.dropna()
    #     display(df_ml)

        # plot the correlations between the different models
        corr = df_ml.corr('spearman')
        # Fill diagonal and upper half with NaNs
        mask = np.zeros_like(corr, dtype=bool)
        mask[np.triu_indices_from(mask)] = True
        corr[mask] = np.nan
        display((corr.style.background_gradient(vmin=0, vmax=1).highlight_null(null_color='#f1f1f1')).format(precision=2))  # Color NaNs grey

        # visualize the subject class between different models
        sort_order = [lbls,*models]

        fig, ax = plt.subplots(figsize=(16,len(sort_order)))

        plot_subject_classes(df_ml,
                             ax, confs=['Sex', 'Site'],
                             sort_order=sort_order, 
                             title='Comparison of Model predictions on the same subject',
                             lw=30, cmap=plt.cm.YlGnBu)
        plt.show()