import pandas as pd 
import numpy as np
from glob import glob
from os.path import join 
import os, sys
from scikits.bootstrap import ci
import matplotlib.pyplot as plt
import seaborn as sns


# load all the labels used in the study as one pandas dataframe along with sex and site and subject ID information
import os, sys, inspect 
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from imagen_dataloader import *

def load_all_labels(raw_vals=False):
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

    final_df = pd.DataFrame(index=Imagen(exclude_holdout=False).df_out.index)

    x_tp = "FU3" #, "FU2", "BL"
    lbl_combos = lbl_combinations
    name = ""
    use_all_data=True
    viz=0
    feature_cols=".+"
    confs=['sex','site']

    for i, (csv, col, c0, c1, colname) in enumerate(lbl_combos):
        if use_all_data:
            d = Imagen(exclude_holdout=False)
        elif use_only_holdout:  
            d = Imagen()
            d.df = d.df_holdout
            d.df_out = pd.DataFrame(index=d.df_holdout.index)
        else:
            d = Imagen()         

        if qs_is_raw_csv[csv]:
            dfq = pd.read_csv(qs[csv], usecols=["User code", col], dtype={"User code":str})
            dfq["ID"] = dfq["User code"].str.replace("-C", "").replace("-I", "").astype(int)
            dfq = dfq.drop("User code", axis=1)
        else:
            dfq = pd.read_csv(qs[csv], usecols=["ID", col])
        
        if raw_vals:
            binarize, class0, class1=False, None, None
        else:
            binarize, class0, class1=True, c0, c1
            
        d.load_label(dfq, col=col, viz=(viz>1), binarize=binarize, class0=class0, class1=class1, y_colname=colname)
        # from x.prepare_X() get the subject IDS to filter out the right subjects
        feature = pd.read_csv(join(d.DATA_DIR, f"IMAGEN_BIDS/sMRI-derivatives_{x_tp}.csv"), index_col="ID")
        # filter columns
        feature = feature.filter(regex=feature_cols)
        d.df_out = d.df_out.loc[feature.index] 
        
        if not raw_vals:
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
    final_df = final_df.dropna()
    
    return final_df



# plot the correlations between the different models
def show_corr(df, corrs=["r-squared"]):
    
    def accuracy(a, b): return (a.astype(int) ==b.astype(int)).sum()*100 / len(b)
    def r_squared(a, b): return np.corrcoef(a, b)[0, 1]**2
    # calculate the correlations
    for corr_type in corrs:
        corr_type = corr_type.lower()
        print(corr_type.title())
        if corr_type in ['pearson', 'kendall', 'spearman']:
            vmin, vmax, cmap = -1, 1, 'RdYlBu_r'
        elif corr_type in ['agreement', "accuracy"]:
            vmin, vmax, cmap = 0, 100, 'PuRd'
            corr_type = accuracy
            print("::(%)")
        elif corr_type in ['r2','r_squared','r-squared']:
            vmin, vmax, cmap = 0, 0.95, 'hot_r'
            corr_type = r_squared
        else:
            print("error! unknown correlation type", corr_type)
        
        corr = df.corr(corr_type)
        # Fill diagonal and upper half with NaNs
        mask = np.zeros_like(corr, dtype=bool)
        mask[np.triu_indices_from(mask)] = True
        corr[mask] = np.nan
        corr = corr.style.background_gradient(
            vmin=vmin, vmax=vmax, cmap=cmap).highlight_null(null_color='#f1f1f1').format('{:.2f}') # Color NaNs grey
        return corr
        
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

###########################################################################
### my version of JiHoon's function todo ##
class Instrument_loader:
    def __init__(self, data_dir="/ritter/share/data/IMAGEN"):
        # Set the directory path: IMAGEN
        self.data_dir = data_dir
        self.INSTRUMENTS = {'NEO':'',
                            'SURPS':'',
                            'CTQ': ''}

    def load_INSTRUMENT(self, instrument_name, session):
        # Set Session, ROI data, and Instrument file
        self.instrument_file = instrument_file
        assert instrument_name in self.VALID_INSTRUMENT_NAMES
        self.instrument_name = instrument_name
        # Load the instrument file       
        instrument_path = f"{self.data_dir}/IMAGEN_RAW/2.7/{self.session}"+\
                            f"/psytools/{self.instrument_file}"
        df = pd.read_csv(instrument_path, low_memory=False)
        # Add the column : ID
        def usercode(x):
            return int(x[:12])
        df['ID'] = df['User code'] if self.session=='FU3' \
        else df['User code'].apply(usercode)
        self.df = df.set_index('ID').reindex(drop=True)

    def generate_new_instrument(self, save=False, viz=False):
        """ Generate the new instrument,
        Rows: Subjects by ID Hdf5 & INSTRUMENT
        Cols: ROI Columns, Data, Session, ID, Sex, Site, Class (HC, AAM)

        Parameters
        ----------   
        save : Boolean, optional
            If it is true, save to *.csv
        
        viz : Boolean, optional
            If it is true, print the steps
        
        Notes
        -----
        This function rename the columns and select the ROI columns:
            Psychological profile:
                NEO, SURPS
            Socio-economic profile:
                CTQ, LEQ, PBQ, CTS, GEN
            Other co-morbidities:
                FTND, DAST, SCID, DMQ, BSI, AUDIT, MAST 
        It may adjust the data type and values.
        
        """
        # ----------------------------------------------------- #
        # Data, Session, ID, Sex, Site, Class columns           #
        # ----------------------------------------------------- #
        df_temp = self.df
        df_temp['Data'] = self.h5py_file[:-3]
        df_temp['Session'] = self.session
        df_temp['ID'] = df_temp.index
        
        # ----------------------------------------------------- #
        # ROI Columns: Psychological profile                    #
        # ----------------------------------------------------- #
        if self.instrument_name == 'NEO':
            # Rename the columns
            new_df = df_temp.rename(
                columns = {
                    "neur_mean":"Neuroticism mean",
                    "extr_mean":"Extroversion mean",
                    "open_mean":"Openness mean",
                    "agre_mean":"Agreeableness mean",
                    "cons_mean":"Conscientiousness mean",
                }
            )
            # Set the roi variables
            self.Variables = [
                'Neuroticism mean','Extroversion mean','Openness mean',
                'Agreeableness mean','Conscientiousness mean',
                'Session','Data','ID','Sex','Site','Class'
            ]
            self.new_df = new_df[self.Variables]
        
        if self.instrument_name == 'SURPS':
            # Rename the columns
            new_df = df_temp.rename(
                columns = {
                    "h_mean":"Hopelessness mean",
                    "as_mean":"Anxiety sensitivity mean",
                    "imp_mean":"Impulsivity mean",
                    "ss_mean":"Sensation seeking mean",
                }
            )
            # Set the roi variables
            self.Variables = [
                'Hopelessness mean','Anxiety sensitivity mean',
                'Impulsivity mean', 'Sensation seeking mean',
                'Data','Session','ID','Sex', 'Site', 'Class'
            ]
            self.new_df = new_df[self.Variables]

        # ----------------------------------------------------- #
        # ROI Columns: Socio-economic profile                   #
        # ----------------------------------------------------- #
        if self.instrument_name == 'CTQ':
            emot_abu = ['CTQ_3','CTQ_8','CTQ_14','CTQ_18','CTQ_25']
            phys_abu = ['CTQ_9','CTQ_11','CTQ_12','CTQ_15','CTQ_17']
            sexual_abu = ['CTQ_20','CTQ_21','CTQ_23','CTQ_24','CTQ_27']
            emot_neg = ['CTQ_5','CTQ_7','CTQ_13','CTQ_19','CTQ_28']
            phys_neg = ['CTQ_1','CTQ_2','CTQ_4','CTQ_6','CTQ_26']
            denial = ['CTQ_10','CTQ_16','CTQ_22']
            # Generate the columns
            df_temp['Emotional abuse sum'] = df_temp[emot_abu].sum(axis=1,skipna=False)
            df_temp['Physical abuse sum'] = df_temp[phys_abu].sum(axis=1,skipna=False)
            df_temp['Sexsual abuse sum'] = df_temp[sexual_abu].sum(axis=1,skipna=False)
            df_temp['Emotional neglect sum'] = df_temp[emot_neg].sum(axis=1,skipna=False)
            df_temp['Physical neglect sum'] = df_temp[phys_neg].sum(axis=1,skipna=False)
            df_temp['Denial sum'] = df_temp[denial].sum(axis=1, skipna=False)
            new_df = df_temp
            # Set the roi variables
            self.Variables = [
                'Emotional abuse sum','Physical abuse sum','Sexsual abuse sum',
                'Emotional neglect sum','Physical neglect sum','Denial sum',
                'Data','Session','ID','Sex','Site','Class'
            ]
            self.new_df = new_df[self.Variables]

        if self.instrument_name == 'LEQ':
            # Rename the columns
            new_df = df_temp.rename(
                columns = {
                    # Mean valence of events
                    "family_valence":"Family valence",
                    "accident_valence":"Accident valence",
                    "sexuality_valence":"Sexuality valence",
                    "autonomy_valence":"Autonomy valence",
                    "devience_valence":"Devience valence",
                    "relocation_valence":"Relocation valence",
                    "distress_valence":"Distress valence",
                    "noscale_valence":"Noscale valence",
                    "overall_valence":"Overall valence",
                    # Mean frequency lifetime
                    "family_ever_meanfreq":"Family mean frequency",
                    "accident_ever_meanfreq":"Accident mean frequency",
                    "sexuality_ever_meanfreq":"Sexuality mean frequency",
                    "autonomy_ever_meanfreq":"Autonomy mean frequency",
                    "devience_ever_meanfreq":"Devience mean frequency",
                    "relocation_ever_meanfreq":"Relocation mean frequency",
                    "distress_ever_meanfreq":"Distress mean frequency",
                    "noscale_ever_meanfreq":"Noscale mean frequency",
                    "overall_ever_meanfreq":"Overall mean frequency",
                }
            )
            # Set the roi variables
            self.Variables = [
                'Family valence','Accident valence','Sexuality valence',
                'Autonomy valence','Devience valence','Relocation valence',
                'Distress valence','Noscale valence','Overall valence',
                'Family mean frequency','Accident mean frequency',
                'Sexuality mean frequency','Autonomy mean frequency',
                'Devience mean frequency','Relocation mean frequency',
                'Distress mean frequency','Noscale mean frequency',
                'Overall mean frequency',
                'Data','Session','ID','Sex','Site','Class']
            self.new_df = new_df[self.Variables]

        if self.instrument_name == 'PBQ':
            # Generate the columns
            def test(x):
                if x == 0: return 'No'
                elif x == 1: return 'Yes'
                elif x == -1: return 'not known'
                elif x == -2: return 'not available'
                else: return np.NaN
            def day(x):
                if x == 2: return 'yes, every day'
                elif x == 1: return 'yes, on occasion'
                elif x == 0: return 'no, not at all'
                elif x == -1: return 'not known'
                elif x == -2: return 'not available'
                else: return np.NaN
            def age(x):
                if x == -1: return 'not known'
                elif x == -2: return 'not available'
                else: return x
            def cigarettes(x):
                if x == 1: return 'Less than 1 cigarette per week'
                elif x == 2: return 'Less than 1 cigarette per day'
                elif x == 3: return '1-5 cigarettes per day'
                elif x == 8: return '6-10 cigarettes per day'
                elif x == 15: return '11-20 cigarettes per day'
                elif x == 25: return '21-30 cigarettes per day'
                elif x == 30: return 'More than 30 cigarettes per day'
                elif x == -1: return 'not known'
                elif x == -2: return 'not available'
                else: return np.NaN
            def alcohol(x):
                if x == 1: return 'Monthly or less'
                elif x == 2: return 'Two to four times a month'
                elif x == 3: return 'Two to three times a week'
                elif x == 4: return 'Four or more times a week'
                elif x == -1: return 'not known'
                elif x == -2: return 'not available'
                else: return np.NaN
            def drinks(x):
                if x == 0: return '1 or 2'
                elif x == 1: return '3 or 4'
                elif x == 2: return '5 or 6'
                elif x == 3: return '7 to 9'
                elif x == 4: return '10 or more'
                elif x == -1: return 'not known'
                elif x == -2: return 'not available'
                else: return np.NaN
            def stage(x):
                if x == 1: return 'first trimester'
                elif x == 2: return 'second trimester'
                elif x == 3: return 'third trimester'
                elif x == 12: return 'first and second'
                elif x == 23: return 'second and third'
                elif x == 13: return 'first and third'
                elif x == 4: return 'whole pregnancy'
                elif x == -1: return 'not known'
                elif x == -2: return 'not available'
                else: return np.NaN
            # Rename the values
            df_temp["pbq_03"] = df_temp['pbq_03'].apply(test)
            df_temp['pbq_03a'] = df_temp['pbq_03a'].apply(day)
            df_temp['pbq_03b'] = df_temp['pbq_03b'].apply(age)
            df_temp["pbq_03c"] = df_temp['pbq_03c'].apply(cigarettes)
            df_temp["pbq_05"] = df_temp['pbq_05'].apply(test)
            df_temp["pbq_05a"] = df_temp['pbq_05a'].apply(cigarettes)
            df_temp["pbq_05b"] = df_temp['pbq_05b'].apply(cigarettes)
            df_temp["pbq_05c"] = df_temp['pbq_05c'].apply(cigarettes)
            df_temp["pbq_06"] = df_temp['pbq_06'].apply(test)
            df_temp["pbq_06a"] = df_temp['pbq_06a'].apply(cigarettes)
            df_temp["pbq_12"] = df_temp['pbq_12'].apply(test)
            df_temp["pbq_13"] = df_temp['pbq_13'].apply(test)
            df_temp["pbq_13a"] = df_temp['pbq_13a'].apply(alcohol)
            df_temp["pbq_13b"] = df_temp['pbq_13b'].apply(drinks)
            df_temp["pbq_13g"] = df_temp['pbq_13g'].apply(stage)    
            # Rename the columns
# It can be rename the columns           
            new_df = df_temp
            # Set the roi variables
            self.Variables = [
                'pbq_03','pbq_03a','pbq_03b','pbq_03c','pbq_05',
                'pbq_05a','pbq_05b','pbq_05c','pbq_06','pbq_06a',
                'pbq_12','pbq_13','pbq_13a','pbq_13b','pbq_13g',
                'Data','Session','ID','Sex', 'Site', 'Class'
            ]
            Check_df = new_df[self.Variables]
            # Exclude the row
            # Duplicate ID: 71766352, 58060181, 15765805, 15765805 in FU1
            # Different ID: 12809392 in both BL and FU1
            for i in [71766352, 58060181, 15765805, 12809392]:
                is_out = (Check_df['ID']==i) & (Check_df['Session']=='FU1')
                Check_df = Check_df[~is_out]
            for i in [12809392]:
                is_out = (Check_df['ID']==i) & (Check_df['Session']=='BL')
                Check_df = Check_df[~is_out]
            self.new_df = Check_df

        if self.instrument_name == 'CTS':
            # Rename the columns
            new_df = df_temp.rename(
                columns={
                    "cts_assault":"Assault mean",
                    "cts_injury":"Injury mean",
                    "cts_negotiation":"Negotiation mean",
                    "cts_psychological_aggression":"Psychological aggression mean",
                    "cts_sexual_coercion":"Sexual coercion mean",
                }
            )
            # Set the roi variables
            self.Variables = [
                'Assault mean','Injury mean','Negotiation mean',
                'Psychological aggression mean','Sexual coercion mean',
                'Data','Session','ID','Sex','Site','Class'
            ]
            self.new_df = new_df[self.Variables]
        
        if self.instrument_name == 'GEN':
            # Generate the columns
            def disorder(x):
                if x == 'ALC': return 'Alcohol problems'
                elif x == 'DRUG': return 'Drug problems'
                elif x == 'SCZ': return 'Schizophrenia'
                elif x == 'SCZAD': return 'Schizoaffective Disorder'
                elif x == 'DPR_R': return 'Major Depression recurrent'
                elif x == 'DPR_SE': return 'Major Depression single episode'
                elif x == 'BIP_I': return 'Bipolar I Disorder'
                elif x == 'BIP_II': return 'Bipolar II Disorder'
                elif x == 'OCD': return 'Obessive-compulsive Disroder'
                elif x == 'ANX': return 'Anxiety Disorder'
                elif x == 'EAT': return 'Eating Disorder'
                elif x == 'SUIC': return 'Suicide / Suicidal Attempt'
                elif x == 'OTHER': return 'Other'
                else: return np.NaN
            # Rename the values
            df_temp['Disorder_PF_1'] = df_temp['Disorder_PF_1'].apply(disorder)
            df_temp['Disorder_PF_2'] = df_temp['Disorder_PF_2'].apply(disorder)
            df_temp['Disorder_PF_3'] = df_temp['Disorder_PF_3'].apply(disorder)
            df_temp['Disorder_PF_4'] = df_temp['Disorder_PF_4'].apply(disorder)
            df_temp['Disorder_PM_1'] = df_temp['Disorder_PM_1'].apply(disorder)
            df_temp['Disorder_PM_2'] = df_temp['Disorder_PM_2'].apply(disorder)
            df_temp['Disorder_PM_3'] = df_temp['Disorder_PM_3'].apply(disorder)
            df_temp['Disorder_PM_4'] = df_temp['Disorder_PM_4'].apply(disorder)
            df_temp['Disorder_PM_5'] = df_temp['Disorder_PM_5'].apply(disorder)
            df_temp['Disorder_PM_6'] = df_temp['Disorder_PM_6'].apply(disorder)
            new_df = df_temp
            Variables = [
                'Disorder_PF_1','Disorder_PF_2','Disorder_PF_3','Disorder_PF_4',
                'Disorder_PM_1','Disorder_PM_2','Disorder_PM_3','Disorder_PM_4',
                'Disorder_PM_5','Disorder_PM_6','Session','ID','Data','Sex','Site','Class'
            ]
            Check_df = new_df[Variables]
            Check_df['Paternal_disorder'] = Check_df.loc[:, Check_df.columns[:4]].apply(
                lambda x: ','.join(x.dropna().astype(str)), axis=1)
            Check_df['Maternal_disorder'] = Check_df.loc[:, Check_df.columns[4:9]].apply(
                lambda x: ','.join(x.dropna().astype(str)), axis=1)

            # Set the roi variables
            self.Variables = [
                'Paternal_disorder','Maternal_disorder',
                'Session','ID','Data','Sex','Site','Class'
            ]
            self.new_df = Check_df[self.Variables]
            
        # ----------------------------------------------------- #
        # ROI Columns: Other co-morbidities                     #
        # ----------------------------------------------------- #
        if self.instrument_name == 'FTND':
            # Generate the columns
            def test(x):
                if (7<=x and x <=10): return 'highly dependent'
                elif (4<=x and x <=6): return 'moderately dependent'
                elif (x<4): return 'less dependent'
                else: return np.NaN
            df_temp["Nicotine dependence"] = df_temp['ftnd_sum'].apply(test)
            new_df = df_temp
            # Set the roi variables
            self.Variables = [
                'Nicotine dependence','Data','Session','ID','Sex','Site','Class'
            ]
            self.new_df = new_df[self.Variables]

#################################################################################
#         if self.instrument_name == 'DAST':
#         def DAST_SESSION(SESSION):
#             if SESSION == 'FU3':
#                 Variables = ['sex', 'site', 'class']
#                 DATA_df = self.new_df[Variables]
#                 return Variables, DATA_df
#             if 'DAST' == self.instrument_name: # 'DAST'
#                 self.VARIABLES, self.new_df2 = DAST_SESSION(self.session)

#         if self.instrument_name == 'SCID':
#         def SCID_SESSION(SESSION):
#             if SESSION == 'FU3':
#                 Variables = ['sex', 'site', 'class']
#                 DATA_df = self.new_df[Variables]
#                 return Variables, DATA_df
#             if 'SCID' == self.instrument_name: # 'SCID'
#                 self.VARIABLES, self.new_df2 = SCID_SESSION(self.session)

#         if self.instrument_name == 'DMQ':
#         def DMQ_SESSION(SESSION):
#             if SESSION == 'FU3':
#                 Variables = ['sex', 'site', 'class']
#                 DATA_df = self.new_df[Variables]
#                 return Variables, DATA_df
#             if 'DMQ' == self.instrument_name: # 'DMQ'
#                 self.VARIABLES, self.new_df2 = DMQ_SESSION(self.session)

#         if self.instrument_name == 'BSI':
#         def BSI_SESSION(SESSION):
#             if SESSION == 'FU3':
#                 ## Somatization
#                 Somatization_labels = ['BSI_02', 'BSI_07', 'BSI_23',
#                                        'BSI_29', 'BSI_30', 'BSI_33', 'BSI_37']
#                 ## Obsession-Compulsion
#                 Obsession_compulsion_labels = ['BSI_05', 'BSI_15', 'BSI_26',
#                                                'BSI_27', 'BSI_32', 'BSI_36']
#                 ## Interpersonal Sensitivity
#                 Interpersonal_sensitivity_labels = ['BSI_20', 'BSI_21',
#                                                     'BSI_22', 'BSI_42']
#                 ## Depression
#                 Depression_labels = ['BSI_09', 'BSI_16', 'BSI_17',
#                                      'BSI_18', 'BSI_35', 'BSI_50']
#                 ## Anxiety
#                 Anxiety_labels = ['BSI_01', 'BSI_12', 'BSI_19',
#                                   'BSI_38', 'BSI_45', 'BSI_49']
#                 ## Hostility
#                 Hostility_labels = ['BSI_06', 'BSI_13', 'BSI_40',
#                                     'BSI_41', 'BSI_46']
#                 ## Phobic Anxiety
#                 Phobic_anxiety_labels = ['BSI_08', 'BSI_28', 'BSI_31',
#                                          'BSI_43', 'BSI_47']
#                 ## Paranoid Ideation
#                 Paranoid_ideation_labels = ['BSI_04', 'BSI_10', 'BSI_24',
#                                             'BSI_48', 'BSI_51']
#                 ## Psychoticism
#                 Psychoticism_labels = ['BSI_03', 'BSI_14', 'BSI_34',
#                                        'BSI_44', 'BSI_53']
#                 df_BSI = self.new_df
#                 df_BSI['somatization_mean'] = df_CTQ[Somatization_labels].mean(axis=1,
#                                                                                skipna=False)
#                 df_BSI['obsession_compulsion_mean'] = df_CTQ[Somatization_labels].mean(axis=1,
#                                                                                skipna=False)             
#         elif 'BSI' == self.instrument_name: # 'BSI'
#             self.VARIABLES, self.new_df2 = BSI_SESSION(self.session)             
#                 Variables = Somatization_labels + Obsession_compulsion_labels \
#                         + Interpersonal_sensitivity_labels + Depression_labels \
#                         + Anxiety_labels + Hostility_labels \
#                         + Phobic_anxiety_labels + Paranoid_ideation_labels \
#                         + Psychoticism_labels + ['sex', 'site', 'class']
#                 DATA_df = self.new_df[Variables]
#                 return Variables, DATA_df

#         if self.instrument_name == 'AUDIT':
#         def AUDIT_SESSION(SESSION):
#             if SESSION == 'FU3':
#                 Variables = ['sex', 'site', 'class']
#                 DATA_df = self.new_df[Variables]
#                 return Variables, DATA_df
#         elif 'AUDIT' == self.instrument_name: # 'AUDIT'
#             self.VARIABLES, self.new_df2 = AUDIT_SESSION(self.session)

#         if self.instrument_name == 'MAST':
#         def MAST_SESSION(SESSION):
#             if SESSION == 'FU3':
#                 Variables = ['sex', 'site', 'class']
#                 DATA_df = self.new_df[Variables]
#                 return Variables, DATA_df
#         elif 'MAST' == self.instrument_name: # 'MAST'
#             self.VARIABLES, self.new_df2 = MAST_SESSION(self.session)  
#         self.new_df = new_df[self.Variables]
#################################################################################
        if save == True:
            phenotype = self.h5py_file.replace(".h5", "")
            save_absolute_path = f"{self.data_dir}/Instrument/"+\
                                 f"{phenotype}_{self.session}_{self.instrument_name}.csv"
            # set the save option
            if not os.path.isdir(os.path.dirname(save_absolute_path)):
                os.makedirs(os.path.dirname(save_absolute_path))
            self.new_df.to_csv(save_absolute_path)
            
        if viz == True:
            print(f"{'-'*83} \n{self.__str__()} \n{'-'*83}")
            print(f"{self.new_df.info(), self.new_df.describe()}")
        return self.new_df
    
    
    def get_instrument(self, h5py_file, SESSION, instrument_file,
                       DATA, save=False, viz=False):
        """ Load the Hdf5, INSTRUMENT. And generate the new instruemnt

        Parameters
        ----------
        h5py_file : string,
            The IMAGEN's h5df file name (*.h5)

        SESSION : string
            One of the four SESSION (BL, FU1, FU2, FU3)
            
        instrument_file : string
            The IMAGEN's instrument file (*.csv)
        
        DATA : string
            The research of interest Instrument Name

        save : Boolean, optional
            If it is true, save to *.csv
        
        viz : Boolean, optional
            If it is true, print the steps

        Returns
        -------        
        self.new_df : pandas.dataframe
            The new instrument file (*.csv)
        
        self.Variables : string
            Instruments columns: ROI Columns, Sex, Site, Class

        Examples
        --------
        >>> from imagen_instrumentloader import *
        >>> NEO = IMAGEN_instrument()
        >>> df_binge_FU3_NEO = NEO.get_instrument(
        ...     "newlbls-fu3-espad-fu3-19a-binge-n650.h5", # h5files
        ...     "FU3",                                     # session
        ...     "IMAGEN-IMGN_NEO_FFI_FU3.csv",             # instrument
        ...     "NEO",                                     # roi name
        ...     save = False,                              # save
        ...     viz = False)                               # summary
        
        """
        self.load_Hdf5(h5py_file)
        self.load_INSTRUMENT(SESSION, DATA, instrument_file)
        self.generate_new_instrument(save, viz)
        return self.new_df