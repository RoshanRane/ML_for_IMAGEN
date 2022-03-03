#################################################################################
#!/usr/bin/env python
# coding: utf-8
""" IMAGEN Post hoc analysis Helper in all Session """
# Author: JiHoon Kim, <jihoon.kim@fu-berlin.de>
#       : Roshan Prakash Rane, <roshan.rane@bccn-berlin.de>
# Last modified: 17th January 2022
import os, sys, inspect 
from os.path import join 
import h5py as h5
import math
import shap
import pickle
import pandas as pd
import numpy as np
from glob import glob
from joblib import load
from sklearn.preprocessing import StandardScaler
from scikits.bootstrap import ci
from sklearn.metrics import *

from IPython.display import display_html 
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm
from matplotlib.colors import LinearSegmentedColormap
from shap.plots import colors
from matplotlib.lines import Line2D
import seaborn as sns
from skimage.segmentation import find_boundaries
from scipy import stats
from scipy.stats import shapiro, levene, ttest_ind, bartlett
from statannot import add_stat_annotation

import nibabel as nib
from nilearn import plotting, image
import warnings


# load all the labels used in the study as one pandas dataframe along with sex and site and subject ID information
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
sys.path.insert(1, '../MLpipelines/')
from plotResults import *
from imagen_dataloader import *

sns.set_style("darkgrid")
warnings.filterwarnings('ignore')

#################################################################################
#                             POST HOC DATA LOADER                              #
#################################################################################

class INSTRUMENT_loader:
    def __init__(self, DATA_DIR="/ritter/share/data/IMAGEN"):
        """ Set up path
        
        Parameters
        ----------
        DATA_DIR : string, optional
            Directory IMAGEN absolute path
        
        """
        # Set the directory path: IMAGEN
        self.DATA_DIR = DATA_DIR
    
    def set_INSTRUMENT(self, DATA, save=False):
        """ Save all session instrument in one file
        
        Parameters
        ----------
        DATA : string,
            instrument name
        save : boolean,
            save the pandas.dataframe to .csv file
            
        Returns
        -------
        DF3 : pandas.dataframe
            instrument in all session (BL, FU1, FU2, FU3)
            
        Examples
        --------
        >>> from imagen_posthocloader import *
        >>> DATA = INSTRUMENT_loader()
        >>> DF3 = DATA.set_INSTRUMENT(
        ...     'NEO',                          # INSTRUMENT
        ...     save = True)                    # Save
        >>> DF_FU3 = DF3.groupby('Session').get_group('FU3')
        
        Notes
        -----
        If only one session has information,
        the same value is copied to all sessions based on ID.
        (e.g. CTQ[FU2], CTS[BL], PBQ[BL,FU1], BSI[FU3], PANAS[FU3])
        
        """
        # ----------------------------------------------------- #
        # ROI Columns: Demographic profile                      #
        # ----------------------------------------------------- #        
        if DATA == "BMI":
            pass
        
        if DATA == "PBQ":
            # Set the files with session and roi columns
            PBQ = [
                ('FU1','IMAGEN-IMGN_PBQ_FU_RC1-BASIC_DIGEST.csv'),
                ('BL', 'IMAGEN-IMGN_PBQ_RC1-BASIC_DIGEST.csv')
            ]
            ROI = [
                'ID','Session','pbq_03','pbq_03a','pbq_03b','pbq_03c',
                'pbq_05','pbq_05a','pbq_05b','pbq_05c','pbq_06','pbq_06a',
                'pbq_12','pbq_13','pbq_13a','pbq_13b','pbq_13g',
            ]
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
            # Generate the instrument files in one dataframe
            PBQ_LIST = []
            for SES, CSV in PBQ:
                path = f"{self.DATA_DIR}/IMAGEN_RAW/2.7/{SES}/psytools/{CSV}"
                DF = pd.read_csv(path, low_memory=False)
                # Rename the values
                DF['ID'] = DF['User code'] if SES=='FU3' else DF['User code'].apply(lambda x : int(x[:12]))
                DF['Session'] = SES
                DF["pbq_03"] = DF['pbq_03'].apply(test)
                DF['pbq_03a'] = DF['pbq_03a'].apply(day)
                DF['pbq_03b'] = DF['pbq_03b'].apply(age)
                DF["pbq_03c"] = DF['pbq_03c'].apply(cigarettes)
                DF["pbq_05"] = DF['pbq_05'].apply(test)
                DF["pbq_05a"] = DF['pbq_05a'].apply(cigarettes)
                DF["pbq_05b"] = DF['pbq_05b'].apply(cigarettes)
                DF["pbq_05c"] = DF['pbq_05c'].apply(cigarettes)
                DF["pbq_06"] = DF['pbq_06'].apply(test)
                DF["pbq_06a"] = DF['pbq_06a'].apply(cigarettes)
                DF["pbq_12"] = DF['pbq_12'].apply(test)
                DF["pbq_13"] = DF['pbq_13'].apply(test)
                DF["pbq_13a"] = DF['pbq_13a'].apply(alcohol)
                DF["pbq_13b"] = DF['pbq_13b'].apply(drinks)
                DF["pbq_13g"] = DF['pbq_13g'].apply(stage)
                DF2 = DF[ROI]
                PBQ_LIST.append(DF2)
            PBQ = pd.concat(PBQ_LIST)
            # Exclude the rows:          
            # Duplicate ID: 71766352, 58060181, 15765805, 15765805 in FU1
            for i in [71766352, 58060181, 15765805, 12809392]:
                is_out = (PBQ['ID']==i) & (PBQ['Session']=='FU1')
                PBQ = PBQ[~is_out]
            # Different ID: 12809392 in both BL and FU1
            for i in [12809392]:
                is_out = (PBQ['ID']==i) & (PBQ['Session']=='BL')
                PBQ = PBQ[~is_out]
            DF3 = PBQ
        
        if DATA == 'GEN':
            # Set the files with session and roi columns
            GEN = [
                ('FU3','IMAGEN-IMGN_GEN_RC5-BASIC_DIGEST.csv'),
                ('FU2','IMAGEN-IMGN_GEN_RC5-BASIC_DIGEST.csv'),
                ('FU1','IMAGEN-IMGN_GEN_RC5-BASIC_DIGEST.csv'),
                ('BL','IMAGEN-IMGN_GEN_RC5-BASIC_DIGEST.csv')
            ]
            ROI = ['ID','Session','Paternal_disorder','Maternal_disorder',
                   'Pd_list','Md_list']
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
            # Generate the instrument files in one dataframe
            GEN_LIST = []
            for SES, CSV in GEN:
                path = f"{self.DATA_DIR}/IMAGEN_RAW/2.7/BL/psytools/{CSV}"
                DF = pd.read_csv(path, low_memory=False)
                DF['ID'] = DF['User code'].apply(lambda x : int(x[:12]))
                DF['Session'] = SES
                DF['Disorder_PF_1'] = DF['Disorder_PF_1'].apply(disorder)
                DF['Disorder_PF_2'] = DF['Disorder_PF_2'].apply(disorder)
                DF['Disorder_PF_3'] = DF['Disorder_PF_3'].apply(disorder)
                DF['Disorder_PF_4'] = DF['Disorder_PF_4'].apply(disorder)
                DF['Disorder_PM_1'] = DF['Disorder_PM_1'].apply(disorder)
                DF['Disorder_PM_2'] = DF['Disorder_PM_2'].apply(disorder)
                DF['Disorder_PM_3'] = DF['Disorder_PM_3'].apply(disorder)
                DF['Disorder_PM_4'] = DF['Disorder_PM_4'].apply(disorder)
                DF['Disorder_PM_5'] = DF['Disorder_PM_5'].apply(disorder)
                DF['Disorder_PM_6'] = DF['Disorder_PM_6'].apply(disorder)
                P1 = DF['Disorder_PF_1'].values.tolist()
                P2 = DF['Disorder_PF_2'].values.tolist()
                P3 = DF['Disorder_PF_3'].values.tolist()
                P4 = DF['Disorder_PF_4'].values.tolist()
                M1 = DF['Disorder_PM_1'].values.tolist()
                M2 = DF['Disorder_PM_1'].values.tolist()
                M3 = DF['Disorder_PM_1'].values.tolist()
                M4 = DF['Disorder_PM_1'].values.tolist()
                M5 = DF['Disorder_PM_1'].values.tolist()
                M6 = DF['Disorder_PM_1'].values.tolist()
                Pd_raw = [list(i) for i in zip(P1,P2,P3,P4)]
                Md_raw = [list(j) for j in zip(M1,M2,M3,M4,M5,M6)]
                DF['Pd_list'] = [[x for x in i if pd.isnull(x) == False] for i in Pd_raw]
                DF['Md_list'] = [[x for x in i if pd.isnull(x) == False] for i in Md_raw]
                Variables = [
                    'ID','Session','Disorder_PF_1','Disorder_PF_2','Disorder_PF_3',
                    'Disorder_PF_4','Disorder_PM_1','Disorder_PM_2','Disorder_PM_3',
                    'Disorder_PM_4','Disorder_PM_5','Disorder_PM_6','Pd_list','Md_list'
                ]
                Check_DF = DF[Variables]
                Check_DF['Paternal_disorder'] = Check_DF.loc[:, Check_DF.columns[2:6]].apply(
                    lambda x: ','.join(x.dropna().astype(str)), axis=1)
                Check_DF['Maternal_disorder'] = Check_DF.loc[:, Check_DF.columns[6:12]].apply(
                    lambda x: ','.join(x.dropna().astype(str)), axis=1)
                DF2 = Check_DF[ROI]
                GEN_LIST.append(DF2)
            GEN = pd.concat(GEN_LIST)
            DF3 = GEN

        if DATA == 'LEQ':
            # Set the files with session and roi columns
            LEQ = [
                ('FU3','IMAGEN-IMGN_LEQ_FU3.csv'),
                ('FU2','IMAGEN-IMGN_LEQ_FU2-IMAGEN_DIGEST.csv'),
                ('FU1','IMAGEN-IMGN_LEQ_FU_RC5-IMAGEN_DIGEST.csv'),
                ('BL' ,'IMAGEN-IMGN_LEQ_RC5-BASIC_DIGEST.csv')
            ]
            ROI = [
                'ID','Session','family_valence','accident_valence','sexuality_valence',
                'autonomy_valence','devience_valence','relocation_valence',
                'distress_valence','noscale_valence','overall_valence',
                'family_ever_meanfreq','accident_ever_meanfreq','sexuality_ever_meanfreq',
                'autonomy_ever_meanfreq','devience_ever_meanfreq','relocation_ever_meanfreq',
                'distress_ever_meanfreq','noscale_ever_meanfreq','overall_ever_meanfreq'
            ]
            # Generate the instrument files in one dataframe
            LEQ_LIST = []
            for SES, CSV in LEQ:
                path = f"{self.DATA_DIR}/IMAGEN_RAW/2.7/{SES}/psytools/{CSV}"
                DF = pd.read_csv(path, low_memory=False)
                DF['ID'] = DF['User code'] if SES=='FU3' else DF['User code'].apply(lambda x : int(x[:12]))
                DF['Session'] = SES
                DF2 = DF[ROI]
                LEQ_LIST.append(DF2)
            LEQ = pd.concat(LEQ_LIST)
            # Rename the columns
            DF3 = LEQ.rename(
                columns = {
                    # Mean valence of events
                    "family_valence"           : "Family valence",
                    "accident_valence"         : "Accident valence",
                    "sexuality_valence"        : "Sexuality valence",
                    "autonomy_valence"         : "Autonomy valence",
                    "devience_valence"         : "Devience valence",
                    "relocation_valence"       : "Relocation valence",
                    "distress_valence"         : "Distress valence",
                    "noscale_valence"          : "Noscale valence",
                    "overall_valence"          : "Overall valence",
                    # Mean frequency lifetime
                    "family_ever_meanfreq"     : "Family mean frequency",
                    "accident_ever_meanfreq"   : "Accident mean frequency",
                    "sexuality_ever_meanfreq"  : "Sexuality mean frequency",
                    "autonomy_ever_meanfreq"   : "Autonomy mean frequency",
                    "devience_ever_meanfreq"   : "Devience mean frequency",
                    "relocation_ever_meanfreq" : "Relocation mean frequency",
                    "distress_ever_meanfreq"   : "Distress mean frequency",
                    "noscale_ever_meanfreq"    : "Noscale mean frequency",
                    "overall_ever_meanfreq"    : "Overall mean frequency",
                }
            )
            
        if DATA == "DAWBA":
            pass
        
        if DATA == "CANTAB":
            pass
        
        # ----------------------------------------------------- #
        # ROI Columns: Psychological profile                    #
        # ----------------------------------------------------- #        
        if DATA == "NEO":
            # Set the files with session and roi columns
            NEO = [
                ('FU3','IMAGEN-IMGN_NEO_FFI_FU3.csv'),
                ('FU2','IMAGEN-IMGN_NEO_FFI_FU2-IMAGEN_SURVEY_DIGEST.csv'),
                ('FU1','IMAGEN-IMGN_NEO_FFI_CHILD_FU_RC5-IMAGEN_SURVEY_DIGEST.csv'),
                ('BL', 'IMAGEN-IMGN_NEO_FFI_CHILD_RC5-IMAGEN_SURVEY_DIGEST.csv')
            ]
            ROI = ['ID','Session','open_mean','cons_mean','extr_mean','agre_mean','neur_mean']
            # Generate the instrument files in one dataframe
            NEO_LIST = []
            for SES, CSV in NEO:
                path = f"{self.DATA_DIR}/IMAGEN_RAW/2.7/{SES}/psytools/{CSV}"
                DF = pd.read_csv(path, low_memory=False)
                DF['ID'] = DF['User code'] if SES=='FU3' else DF['User code'].apply(lambda x : int(x[:12]))
                DF['Session'] = SES
                DF2 = DF[ROI]
                NEO_LIST.append(DF2)
            NEO = pd.concat(NEO_LIST)
            # Rename the columns
            DF3 = NEO.rename(
                columns = {
                    "neur_mean" : "Neuroticism mean",
                    "extr_mean" : "Extroversion mean",
                    "open_mean" : "Openness mean",
                    "agre_mean" : "Agreeableness mean",
                    "cons_mean" : "Conscientiousness mean",
                }
            )

        if DATA == "SURPS":
            # Set the files with session and roi columns
            SURPS = [
                ('FU3','IMAGEN-IMGN_SURPS_FU3.csv'),
                ('FU2','IMAGEN-IMGN_SURPS_FU2-IMAGEN_SURVEY_DIGEST.csv'),
                ('FU1','IMAGEN-IMGN_SURPS_FU_RC5-IMAGEN_SURVEY_DIGEST.csv'),
                ('BL', 'IMAGEN-IMGN_SURPS_RC5-IMAGEN_SURVEY_DIGEST.csv')
            ]
            ROI = ['ID', 'Session', 'as_mean', 'h_mean', 'imp_mean', 'ss_mean']
            # Generate the instrument files in one dataframe
            SURPS_LIST = []
            for SES, CSV in SURPS:
                path = f"{self.DATA_DIR}/IMAGEN_RAW/2.7/{SES}/psytools/{CSV}"
                DF = pd.read_csv(path, low_memory=False)
                DF['ID'] = DF['User code'] if SES=='FU3' else DF['User code'].apply(lambda x : int(x[:12]))
                DF['Session'] = SES
                DF2 = DF[ROI]
                SURPS_LIST.append(DF2)
            SURPS = pd.concat(SURPS_LIST)
            # Rename the columns
            DF3 = SURPS.rename(
                columns = {
                    "as_mean" : "Anxiety Sensitivity mean",
                    "h_mean"  : "Hopelessness mean",
                    "imp_mean": "Impulsivity mean",
                    "ss_mean" : "Sensation seeking mean",
                }
            )
            
        if DATA == "TCI":
            # Set the files with session and roi columns
            TCI = [
                ('FU3','IMAGEN-IMGN_TCI_FU3.csv'),
                ('FU2','IMAGEN-IMGN_TCI_CHILD_FU2-IMAGEN_DIGEST.csv'),
                ('FU1','IMAGEN-IMGN_TCI_CHILD_FU_RC5-IMAGEN_DIGEST.csv'),
                ('BL','IMAGEN-IMGN_TCI_CHILD_RC5-IMAGEN_DIGEST.csv')
            ]
            ROI = ['ID','Session','tci_excit','tci_imp','tci_extra','tci_diso','tci_novseek']
            # Generate the instrument files in one dataframe
            TCI_LIST = []
            for SES, CSV in TCI:
                path = f"{self.DATA_DIR}/IMAGEN_RAW/2.7/{SES}/psytools/{CSV}"
                DF = pd.read_csv(path, low_memory=False)
                DF['ID'] = DF['User code'] if SES=='FU3' else DF['User code'].apply(lambda x : int(x[:12]))
                DF['Session'] = SES
                DF2 = DF[ROI]
                TCI_LIST.append(DF2)
            TCI = pd.concat(TCI_LIST)
            # Rename the columns
            DF3 = TCI.rename(
                columns = {
                    "tci_excit" : "Exploratory excitability vs. Stoic rigidity",
                    "tci_imp"  : "Impulsiveness vs. Reflection",
                    "tci_extra": "Extravagance vs. Reserve",
                    "tci_diso" : "Disorderliness vs. Regimentation",
                    "tci_novseek" : "Total Novelty Seeking score"
                }
            )
            
        if DATA == 'BSI':
            # Set the files with session and roi columns
            BSI = [
                ('FU3','IMAGEN-IMGN_BSI_FU3.csv'),
                ('FU2','IMAGEN-IMGN_BSI_FU3.csv'),
                ('FU1','IMAGEN-IMGN_BSI_FU3.csv'),
                ('BL','IMAGEN-IMGN_BSI_FU3.csv')
            ]
            ROI = ['ID','Session','Somatization mean','Obsession-Compulsion mean','Interpersonal Sensitivity mean',
                   'Depression mean','Anxiety mean','Hostility mean','Phobic Anxiety mean','Paranoid Ideation mean',
                   'Psychoticism mean','Positive Symptom Distress Index','Global Severity Index']
            # Generate the columns
            ## Somatization
            Somatization = ['BSI_02','BSI_07','BSI_23','BSI_29','BSI_30','BSI_33','BSI_37']
            ## Obsession-Compulsion
            Obsession_compulsion = ['BSI_05','BSI_15','BSI_26','BSI_27','BSI_32','BSI_36']
            ## Interpersonal Sensitivity
            Interpersonal_sensitivity = ['BSI_20','BSI_21','BSI_22','BSI_42']
            ## Depression
            Depression = ['BSI_09','BSI_16','BSI_17','BSI_18','BSI_35','BSI_50']
            ## Anxiety
            Anxiety = ['BSI_01','BSI_12','BSI_19','BSI_38','BSI_45','BSI_49']
            ## Hostility
            Hostility = ['BSI_06','BSI_13','BSI_40','BSI_41','BSI_46']
            ## Phobic Anxiety
            Phobic_anxiety = ['BSI_08','BSI_28','BSI_31','BSI_43','BSI_47']
            ## Paranoid Ideation
            Paranoid_ideation = ['BSI_04','BSI_10','BSI_24','BSI_48','BSI_51']
            ## Psychoticism
            Psychoticism = ['BSI_03','BSI_14','BSI_34','BSI_44','BSI_53']
            BSI_labels = ['ID','Session']+Somatization+Obsession_compulsion+\
                         Interpersonal_sensitivity+Depression+Anxiety+\
                         Hostility+Phobic_anxiety+Paranoid_ideation+Psychoticism
            
            # Generate the instrument files in one dataframe
            BSI_LIST = []
            for SES, CSV in BSI:
                path = f"{self.DATA_DIR}/IMAGEN_RAW/2.7/FU3/psytools/{CSV}"
                DF = pd.read_csv(path, low_memory=False)
                DF['ID'] = DF['User code']
                DF['Session'] = SES
                # Rename the values
                DF2 = DF[BSI_labels].replace(
                    {'0': 0, '1': 1, '2': 2,'3': 3, '4': 4, '5': 5,'6': 6,'R':np.NaN}
                )
                DF2 = DF2.dropna()
                DF2['Somatization mean'] = DF2[Somatization].mean(axis=1,skipna=False)
                DF2['Obsession-Compulsion mean'] = DF2[Obsession_compulsion].mean(axis=1,skipna=False)
                DF2['Interpersonal Sensitivity mean'] = DF2[Interpersonal_sensitivity].mean(axis=1,skipna=False)
                DF2['Depression mean'] = DF2[Depression].mean(axis=1,skipna=False)
                DF2['Anxiety mean'] = DF2[Anxiety].mean(axis=1,skipna=False)
                DF2['Hostility mean'] = DF2[Hostility].mean(axis=1,skipna=False)
                DF2['Phobic Anxiety mean'] = DF2[Phobic_anxiety].mean(axis=1,skipna=False)
                DF2['Paranoid Ideation mean'] = DF2[Paranoid_ideation].mean(axis=1,skipna=False)
                DF2['Psychoticism mean'] = DF2[Psychoticism].mean(axis=1,skipna=False)
                DF2['Positive Symptom Distress Index'] = DF2[BSI_labels[2:]].sum(axis=1,skipna=False)
                DF2['Global Severity Index'] = DF2[BSI_labels[2:]].mean(axis=1,skipna=False)
                DF2 = DF2[ROI]
                BSI_LIST.append(DF2)
            BSI = pd.concat(BSI_LIST)
            DF3 = BSI
        
        if DATA == "KIRBY":
            # Set the files with session and roi columns
            pass
        
        if DATA == "BIS":
            pass
        
        if DATA == "CSI":
            pass
        
        if DATA == "PHQ":
            pass
        
        if DATA == "CES":
            pass
        
        if DATA == "ANXDX":
            pass
        
        if DATA == "CAPE":
            pass
        
        if DATA == "SDQ":
            pass
        
        if DATA == "IRI":
            pass
        
        if DATA == "RRS":
            pass
        
        if DATA == "PALP":
            pass
        
        # ----------------------------------------------------- #
        # ROI Columns: Sociial profile                          #
        # ----------------------------------------------------- #
#         if DATA == "CTQ":
#             # Set the files with session and roi columns
#             CTQ = [
#                 ('FU3','IMAGEN-IMGN_CTQ_CHILD_FU2-IMAGEN_DIGEST.csv'),
#                 ('FU2','IMAGEN-IMGN_CTQ_CHILD_FU2-IMAGEN_DIGEST.csv'),
#                 ('FU1','IMAGEN-IMGN_CTQ_CHILD_FU2-IMAGEN_DIGEST.csv'),
#                 ('BL', 'IMAGEN-IMGN_CTQ_CHILD_FU2-IMAGEN_DIGEST.csv')
#             ]
#             ROI = ['ID','Session','ea_sum','pa_sum','sa_sum','en_sum','pn_sum','dn_sum']
#             # Set the columns for computation
#             emot_abu = ['CTQ_3','CTQ_8','CTQ_14','CTQ_18','CTQ_25']
#             phys_abu = ['CTQ_9','CTQ_11','CTQ_12','CTQ_15','CTQ_17']
#             sexual_abu = ['CTQ_20','CTQ_21','CTQ_23','CTQ_24','CTQ_27']
#             emot_neg = ['CTQ_5','CTQ_7','CTQ_13','CTQ_19','CTQ_28']
#             phys_neg = ['CTQ_1','CTQ_2','CTQ_4','CTQ_6','CTQ_26']
#             denial = ['CTQ_10','CTQ_16','CTQ_22']
#             # Generate the instrument files in one dataframe
#             CTQ_LIST = []
#             for SES, CSV in CTQ:
#                 path = f"{self.DATA_DIR}/IMAGEN_RAW/2.7/FU2/psytools/{CSV}"
#                 DF = pd.read_csv(path, low_memory=False)
#                 DF['ID'] = DF['User code'].apply(lambda x : int(x[:12]))
#                 DF['Session'] = SES
#                 DF['ea_sum'] = DF[emot_abu].sum(axis=1,skipna=False)
#                 DF['pa_sum'] = DF[phys_abu].sum(axis=1,skipna=False)
#                 DF['sa_sum'] = DF[sexual_abu].sum(axis=1,skipna=False)
#                 DF['en_sum'] = DF[emot_neg].sum(axis=1,skipna=False)
#                 DF['pn_sum'] = DF[phys_neg].sum(axis=1,skipna=False)
#                 DF['dn_sum'] = DF[denial].sum(axis=1, skipna=False)
#                 DF2 = DF[ROI]
#                 CTQ_LIST.append(DF2)
#             CTQ = pd.concat(CTQ_LIST)
#             # Rename the columns
#             DF3 = CTQ.rename(
#                 columns = {
#                     "ea_sum" : "Emotional abuse sum",
#                     "pa_sum" : "Physical abuse sum",
#                     "sa_sum" : "Sexual abuse sum",
#                     "en_sum" : "Emotional neglect sum",
#                     "pn_sum" : "Physical neglect sum",
#                     "dn_sum" : "Denial sum"
#                 }
#             )
            
        if DATA == "CTQ_MD":
            # Set the files with session and roi columns
            CTQ = [
                ('FU3','IMAGEN-IMGN_CTQ_CHILD_FU2-IMAGEN_DIGEST.csv'),
                ('FU2','IMAGEN-IMGN_CTQ_CHILD_FU2-IMAGEN_DIGEST.csv'),
                ('FU1','IMAGEN-IMGN_CTQ_CHILD_FU2-IMAGEN_DIGEST.csv'),
                ('BL', 'IMAGEN-IMGN_CTQ_CHILD_FU2-IMAGEN_DIGEST.csv')
            ]
            ROI = ['ID','Session','ea_sum','pa_sum','sa_sum','en_sum','pn_sum',
                   'dn_sum','md1','md2','md3']
            # Set the columns for computation
            emot_abu = ['CTQ_3','CTQ_8','CTQ_14','CTQ_18','CTQ_25']
            phys_abu = ['CTQ_9','CTQ_11','CTQ_12','CTQ_15','CTQ_17']
            sexual_abu = ['CTQ_20','CTQ_21','CTQ_23','CTQ_24','CTQ_27']
            emot_neg = ['CTQ_5','CTQ_7','CTQ_13','CTQ_19','CTQ_28']
            phys_neg = ['CTQ_1','CTQ_2','CTQ_4','CTQ_6','CTQ_26']
            denial = ['CTQ_10','CTQ_16','CTQ_22']
            # Generate the instrument files in one dataframe
            CTQ_LIST = []
            for SES, CSV in CTQ:
                path = f"{self.DATA_DIR}/IMAGEN_RAW/2.7/FU2/psytools/{CSV}"
                DF = pd.read_csv(path, low_memory=False)
                DF['ID'] = DF['User code'].apply(lambda x : int(x[:12]))
                DF['Session'] = SES
                DF['ea_sum'] = DF[emot_abu].sum(axis=1,skipna=False)
                DF['pa_sum'] = DF[phys_abu].sum(axis=1,skipna=False)
                DF['sa_sum'] = DF[sexual_abu].sum(axis=1,skipna=False)
                DF['en_sum'] = DF[emot_neg].sum(axis=1,skipna=False)
                DF['pn_sum'] = DF[phys_neg].sum(axis=1,skipna=False)
                DF['dn_sum'] = DF[denial].sum(axis=1, skipna=False)
                DF['md1'] = DF['CTQ_10']
                DF['md2'] = DF['CTQ_16']
                DF['md3'] = DF['CTQ_22']
                DF2 = DF[ROI]
                CTQ_LIST.append(DF2)
            CTQ = pd.concat(CTQ_LIST)
            # Rename the columns
            DF3 = CTQ.rename(
                columns = {
                    "ea_sum" : "Emotional abuse sum",
                    "pa_sum" : "Physical abuse sum",
                    "sa_sum" : "Sexual abuse sum",
                    "en_sum" : "Emotional neglect sum",
                    "pn_sum" : "Physical neglect sum",
                    "dn_sum" : "Denial sum",
                    "md1" : "MD 1",
                    "md2" : "MD 2",
                    "md3" : "MD 3"
                }
            )

        if DATA == "CTS":
            # Set the files with session and roi columns
            CTS = [
                ('FU3','IMAGEN-IMGN_CTS_PARENT_RC5-BASIC_DIGEST.csv'),
                ('FU2','IMAGEN-IMGN_CTS_PARENT_RC5-BASIC_DIGEST.csv'),
                ('FU1','IMAGEN-IMGN_CTS_PARENT_RC5-BASIC_DIGEST.csv'),
                ('BL', 'IMAGEN-IMGN_CTS_PARENT_RC5-BASIC_DIGEST.csv')
            ]
            ROI = [
                'ID','Session','cts_assault','cts_injury','cts_negotiation',
                'cts_psychological_aggression','cts_sexual_coercion'
            ]
            # Generate the instrument files in one dataframe
            CTS_LIST = []
            for SES, CSV in CTS:
                path = f"{self.DATA_DIR}/IMAGEN_RAW/2.7/BL/psytools/{CSV}"
                DF = pd.read_csv(path, low_memory=False)
                DF['ID'] = DF['User code'].apply(lambda x : int(x[:12]))
                DF['Session'] = SES
                DF2 = DF[ROI]
                CTS_LIST.append(DF2)
            CTS = pd.concat(CTS_LIST)
            # Rename the columns
            DF3 = CTS.rename(
                columns = {
                    "cts_assault"                  : "Assault mean",
                    "cts_injury"                   : "Injury mean",
                    "cts_negotiation"              : "Negotiation mean",
                    "cts_psychological_aggression" : "Psychological Aggression mean",
                    "cts_sexual_coercion"          : "Sexual Coercion mean"
                }
            )
            
        if DATA == "PANAS":
            # Set the files with session and roi columns
            PANAS = [
                ('FU3','IMAGEN-IMGN_PANAS_FU3.csv'),
                ('FU2','IMAGEN-IMGN_PANAS_FU3.csv'),
                ('FU1','IMAGEN-IMGN_PANAS_FU3.csv'),
                ('BL','IMAGEN-IMGN_PANAS_FU3.csv')
            ]
            ROI = ['ID','Session','Positive Affect Score','Negative Affect Score']
            # Set the columns for computation
            panas_pas = ['PANAS_01','PANAS_03','PANAS_05','PANAS_09','PANAS_10',
                         'PANAS_12','PANAS_14','PANAS_16','PANAS_17','PANAS_19']
            panas_nas = ['PANAS_02','PANAS_04','PANAS_06','PANAS_07','PANAS_08',
                         'PANAS_11','PANAS_13','PANAS_15','PANAS_18','PANAS_20']
            # Generate the instrument files in one dataframe
            PANAS_LIST=[]
            for SES, CSV in PANAS:
                path = f"{self.DATA_DIR}/IMAGEN_RAW/2.7/FU3/psytools/{CSV}"
                DF = pd.read_csv(path, low_memory=False)
                DF['ID'] = DF['User code']
                DF['Session'] = SES
                DF['Positive Affect Score'] = DF[panas_pas].sum(axis=1,skipna=False)
                DF['Negative Affect Score'] = DF[panas_nas].sum(axis=1,skipna=False)
                DF2 = DF[ROI]
                PANAS_LIST.append(DF2)
            PANAS = pd.concat(PANAS_LIST)
            DF3 = PANAS
        
        if DATA == "MINI5":
            pass

        # ----------------------------------------------------- #
        # ROI Columns: Substance use profile                    #
        # ----------------------------------------------------- #
        if DATA == "MAST":
            # Set the files with session and roi columns
            MAST = [
                ('FU3','IMAGEN-IMGN_MAST_FU3.csv'),
                ('FU2','IMAGEN-IMGN_MAST_CHILD_FU2-IMAGEN_DIGEST.csv'),
                ('FU1','IMAGEN-IMGN_MAST_PARENT_FU_RC5-BASIC_DIGEST.csv'),
                ('BL','IMAGEN-IMGN_MAST_PARENT_RC5-BASIC_DIGEST.csv')
            ]
            ROI = ['ID','Session','MAST flag','MAST total','MAST Alcohol dependency symptoms','MAST sum']
            # Generate the columns
            def flag(x):
                if (x > 4): return 'positive alchololism screening'
                else: return 'negative alchololism screening'
            # Generate the instrument files in one dataframe
            MAST_LIST = []
            for SES, CSV in MAST:
                path = f"{self.DATA_DIR}/IMAGEN_RAW/2.7/{SES}/psytools/{CSV}"
                DF = pd.read_csv(path, low_memory=False)
                DF['ID'] = DF['User code'] if SES=='FU3' else DF['User code'].apply(lambda x : int(x[:12]))
                DF['Session'] = SES
                # Renmae the values
                DF['MAST total'] = DF['mast_total']
                DF['MAST Alcohol dependency symptoms'] = DF['mast_dsm']
                DF['MAST sum'] = DF['mast_sum']
                DF['MAST flag'] = DF['mast_sum'].apply(flag)
                DF2 = DF[ROI]
                MAST_LIST.append(DF2)
            MAST = pd.concat(MAST_LIST)
            DF3 = MAST

        if DATA == "FTND":
            # Set the files with session and roi columns
            FTND = [
                ('FU3','IMAGEN-IMGN_ESPAD_FU3.csv'),
                ('FU2','IMAGEN-IMGN_ESPAD_CHILD_FU2-IMAGEN_DIGEST.csv'),
                ('FU1','IMAGEN-IMGN_ESPAD_CHILD_FU_RC5-IMAGEN_DIGEST.csv'),
                ('BL','IMAGEN-IMGN_ESPAD_CHILD_RC5-IMAGEN_DIGEST.csv')
            ]
            ROI = ['ID','Session','Likelihood of nicotine dependence child','FTND Sum']
            # Generate the columns
            def test(x):
                if (7<=x and x <=10): return 'highly dependent'
                elif (4<=x and x <=6): return 'moderately dependent'
                elif (x<4): return 'less dependent'
                else: return np.NaN
            # Generate the instrument files in one dataframe
            FTND_LIST = []
            for SES, CSV in FTND:
                path = f"{self.DATA_DIR}/IMAGEN_RAW/2.7/{SES}/psytools/{CSV}"
                DF = pd.read_csv(path, low_memory=False)
                DF['ID'] = DF['User code'] if SES=='FU3' else DF['User code'].apply(lambda x : int(x[:12]))
                DF['Session'] = SES
                # Rename the values
                DF['Likelihood of nicotine dependence child'] = DF['ftnd_sum'].apply(test)
                DF['FTND Sum'] = DF['ftnd_sum']
                DF2 = DF[ROI]
                FTND_LIST.append(DF2)
            FTND = pd.concat(FTND_LIST)
            DF3 = FTND

        if DATA == "DAST":
            # Set the files with session and roi columns
            DAST = [
                
            ]
            # Generate the columns
            
            # Generate the instrument files in one dataframe
#             if SESSION == 'FU3':
#                 Variables = ['sex', 'site', 'class']
#                 DATA_DF = self.NEW_DF[Variables]
#                 return Variables, DATA_DF
#             if 'DAST' == self.DATA: # 'DAST'
#                 self.VARIABLES, self.NEW_DF2 = DAST_SESSION(self.SESSION)
            pass

        if DATA == "SCID":
            # Set the files with session and roi columns
            
            # Generate the columns
            
            # Generate the instrument files in one dataframe
#         def SCID_SESSION(SESSION):
#             if SESSION == 'FU3':
#                 Variables = ['sex', 'site', 'class']
#                 DATA_DF = self.NEW_DF[Variables]
#                 return Variables, DATA_DF
#             if 'SCID' == self.DATA: # 'SCID'
#                 self.VARIABLES, self.NEW_DF2 = SCID_SESSION(self.SESSION)
            pass

        if DATA == "DMQ":
            # Set the files with session and roi columns
            
            # Generate the columns
            
            # Generate the instrument files in one dataframe
#             if SESSION == 'FU3':
#                 Variables = ['sex', 'site', 'class']
#                 DATA_DF = self.NEW_DF[Variables]
#                 return Variables, DATA_DF
#             if 'DMQ' == self.DATA: # 'DMQ'
#                 self.VARIABLES, self.NEW_DF2 = DMQ_SESSION(self.SESSION)
            pass

        if DATA == "BULLY":
            pass

        if DATA == "ESPAD":
            pass
        
        if DATA == "TLFB":
            pass

        if DATA == "AUDIT":
            # Set the files with session and roi columns
            
            # Generate the columns
            
            # Generate the instrument files in one dataframe
#             if SESSION == 'FU3':
#                 Variables = ['sex', 'site', 'class']
#                 DATA_DF = self.NEW_DF[Variables]
#                 return Variables, DATA_DF
#         elif 'AUDIT' == self.DATA: # 'AUDIT'
#             self.VARIABLES, self.NEW_DF2 = AUDIT_SESSION(self.SESSION)
            pass

        if save == True:
            save_path = f"{self.DATA_DIR}/posthoc/all_{DATA}.csv"
            # set the save option
            if not os.path.isdir(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            DF3.to_csv(save_path, index=None)
        return DF3

    def get_INSTRUMENT(self, instrument_file):
        """ Load the INSTRUMENT file
        
        Parameters
        ----------            
        instrument_file : string
            The IMAGEN's instrument file (*.csv)

        Returns
        -------
        DF : pandas.dataframe
            The Instrument dataframe
        
        Notes
        -----
        This function select the ROI:
        Demographic profile - LEQ, PBQ, GEN, BMI, 
        Psychological profile - NEO, SURPS,
        Social profile - CTQ, CTS, 
        Substance use profile - FTND, DAST, SCID, DMQ, BSI, AUDIT, MAST

        Examples
        --------
        >>> from imagen_posthocloader import *
        >>> DATA = IMAGEN_INSTRUMENT()
        >>> DF = DATA.get_INSTRUMENT(
        ...     instrument_file)               # INSTRUMENT
        >>> DF_FU3 = DF.groupby('Session').get_group('FU3')

        """
        # Load the instrument file       
        instrument_path = f"{self.DATA_DIR}/posthoc/{instrument_file}"
        DF = pd.read_csv(instrument_path, low_memory=False)
        return DF

#     def __str__(self):
#         """ Print the instrument loader steps """
#         return "Step 1. load the instrument: " \
#                + "\n        File = " + str(self.instrument_path) \
#                + "\n        The dataset contains " + str(self.DF.shape[0]) \
#                + " samples and " + str(self.DF.shape[1]) + " columns" \
#                + "\n        Variables = " + str(self.VARIABLES)
#             print(f"{'-'*83} \n{self.__str__()} \n{'-'*83}")
#             print(f"{self.NEW_DF.info(), self.NEW_DF.describe()}")

class HDF5_loader:
    def __init__(self, DATA_DIR="/ritter/share/data/IMAGEN"):
        """ Set up path
        
        Parameters
        ----------
        DATA_DIR : string, optional
            Directory IMAGEN absolute path
        
        """
        # Set the directory path: IMAGEN
        self.DATA_DIR = DATA_DIR    

    def set_HDF5(self, DATA, save=False):
        """ Save all session y in one file
        
        Parameters
        ----------
        DATA : string,
            y name
        save : boolean,
            save the pandas.dataframe to .csv file
            
        Returns
        -------
        DF3 : pandas.dataframe
            instrument in all session (BL, FU1, FU2, FU3)

        Examples
        --------
        >>> from imagen_posthocloader import *
        >>> DATA = HDF5_loader()
        >>> DF3 = DATA.set_HDF5(
        ...     DATA,                                 # HDF5
        ...     save = True)                          # save
        >>> DF_FU3 = DF3.groupby('Session').get_group('FU3')

        Notes
        -----
        There are no session in FU2 for imaging file
        y = {Binge} # Other y can be added as if
        
        """
        if DATA == "Binge":
            # Set the files with session and roi columns
            BINGE = [
                ('FU3','Training','newlbls-clean-fu3-espad-fu3-19a-binge-n650.h5'),
                ('FU3','Holdout', 'newholdout-clean-fu3-espad-fu3-19a-binge-n102.h5'),
                ('FU2','Training','newlbls-clean-fu2-espad-fu3-19a-binge-n634.h5'),
                ('FU2','Holdout', 'newholdout-clean-fu2-espad-fu3-19a-binge-n102.h5'),
                ('BL', 'Training','newlbls-clean-bl-espad-fu3-19a-binge-n620.h5'),
                ('BL', 'Holdout', 'newholdout-clean-bl-espad-fu3-19a-binge-n102.h5')
            ]
            ROI = ['ID','Session','y','Dataset','Sex','Site','Class']
            # Generate the instrument files in one dataframe
            BINGE_LIST = []
            for SES, DATASET, HDF5 in BINGE:
                path = f"{self.DATA_DIR}/h5files/{HDF5}"
                # Convert HDF5 to List
                d = h5.File(path,'r')
                # Set All, HC, and AAM
                b_list = list(np.array(d[list(d.keys())[0]]))
                ALL = list(np.array(d['i']))
                HC = [ALL[i] for i, j in enumerate(b_list) if j%2==0]
                AAM = [ALL[i] for i, j in enumerate(b_list) if j%2==1]
                # Set Sex
                sex = list(np.array(d['sex']))
                SEX = ['Male' if i==0 else 'Female' for i in sex]
                # Set Site
                sites = list(np.array(d['site']))
                center = {0: 'Paris', 1: 'Nottingham', 2:'Mannheim', 3:'London',
                          4: 'Hamburg', 5: 'Dublin', 6:'Dresden', 7:'Berlin'}
                SITE = [center[i] for i in sites]
                # Set Class
                target = list(np.array(d[list(d.keys())[0]]))
                CLASS = ['HC' if i==0 else 'AAM' for i in target]
                # Generate the DF
                DF2 = pd.DataFrame(
                    {"ID" : ALL,
                    "Session" : SES,
                    "y" : list(d.keys())[0],
                    "Dataset" : DATASET,
                    "Sex" : SEX,
                    "Site" : SITE,
                    "Class" : CLASS}
                )
                BINGE_LIST.append(DF2)
            DF3 = pd.concat(BINGE_LIST)

        if save == True:
            save_path = f"{self.DATA_DIR}/posthoc/all_{DATA}.csv"
            # set the save option
            if not os.path.isdir(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            DF3.to_csv(save_path, index=None)
        return DF3
    
    def get_HDF5(self, hdf5_file):
        """ Select the ROI y as file
        
        Parameters
        ----------            
        h5df_file : string
            The IMAGEN's instrument file (*.csv)

        Returns
        -------
        DF : pandas.dataframe
            The Instrument dataframe
        
        Notes
        -----
        There are no session in FU2 for imaging file
        y = {Binge} # Other y can be added as if

        Examples
        --------
        >>> from imagen_posthocloader import *
        >>> DATA = HDF5_loader()
        >>> DF = DATA.get_HDF5(
        ...     hdf5_file)                           # HDF5
        >>> DF_FU3 = DF.groupby('Session').get_group('FU3')
        
        """
        # Load the hdf5 file       
        hdf5_path = f"{self.DATA_DIR}/posthoc/{hdf5_file}"
        DF = pd.read_csv(hdf5_path, low_memory=False)
        return DF

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
            at least contain y, ID, numpy.ndarray or other Group mask
            
        Examples
        --------
        >>> from imagen_posthocloader import *
        >>> DATA = HDF5_loader()
        >>> tr_X, tr_X_col_names, tr_Other = DATA.get_train_data(
        ...     'H5_DIR')                                  # DATA
        
        """
        data = h5.File(self.DATA_DIR+"/h5files/"+H5_DIR, 'r')
        print(data.keys(), data.attrs.keys())
        X = data['X'][()]
        X_col = data.attrs['X_col_names']
        X_col_names = np.array([i.replace(")","") for i in X_col])
        self.tr_X = X
        self.tr_X_col_names = X_col_names
        
        y = data[data.attrs['labels'][0]][()]
        ID = data['i'][()]
        
        if group == True:
            sex_mask = data['sex'].astype(bool)[()]
            class_mask = data['Binge'][()].astype(bool)
            self.tr_Other = [y, ID, sex_mask, class_mask]
        else:
            self.tr_Other = [y, ID]
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
            at least contain y, ID, numpy.ndarray or other Group mask
            
        Examples
        --------
        >>> from imagen_posthocloader import *
        >>> DATA = HDF5_loader()
        >>> ho_X, ho_X_col_names, ho_Other = DATA.get_train_data(
        ...     'H5_DIR')                                  # DATA
        
        """
        data = h5.File(self.DATA_DIR+"/h5files/"+H5_DIR, 'r')
#         print(data.keys(), data.attrs.keys())
        X = data['X'][()]
        X_col = data.attrs['X_col_names']
        X_col_names = np.array([i.replace(")","") for i in X_col])
        self.ho_X = X
        self.ho_X_col_names = X_col_names
        
        y = data[data.attrs['labels'][0]][()]
        ID = data['i'][()]
        
        if group == True:
            sex_mask = data['sex'][()]
            class_mask = data['Binge'][()]
            self.ho_Other = [y, ID, sex_mask, class_mask]
        else:
            self.ho_Other = [y, ID]
        X.shape, len(X_col_names)
        return self.ho_X, self.ho_X_col_names, self.ho_Other
    
#     def __str__(self):
#         pass
    
class RUN_loader:
    def __init__(self, DATA_DIR="/ritter/share/data/IMAGEN"):
        """ Set up path
        
        Parameters
        ----------
        DATA_DIR : string, optional
            Directory IMAGEN absolute path
        
        """
        # Set the directory path: IMAGEN
        self.DATA_DIR = DATA_DIR
    
    def set_RUN(self, run_file, save=False):
        """ Save the ML RUN result in one file &
        Generate the RUN classification report for posthoc analysis
        
        Parameters
        ----------
        run_file : string
            ML models result run.csv path
        save : boolean
            if save == True, then save it as .csv
        
        Returns
        -------
        DF3 : pandas.dataframe
            The RUN dataframe
        
        Examples
        --------
        >>> from imagen_posthocloader import *
        >>> DATA = IMAGEN_instrument()
        >>> DF3 = DATA.set_RUN(
        ...     run_file)                              # RUN
        >>> DF_FU3 = DF3.groupby('Session').get_group('fu3')
        
        """
        df = pd.read_csv(run_file, low_memory = False)
        
        DF = []
        for i in range(len(df)):
            test_ids = eval(df['test_ids'].values[i])
            test_lbls = eval(df['test_lbls'].values[i])
            test_probs = [probs[1] for probs in eval(df['test_probs'].values[i])]
            holdout_ids = eval(df['holdout_ids'].values[i])
            holdout_lbls = eval(df['holdout_lbls'].values[i])
            holdout_preds = [probs[1] for probs in eval(df['holdout_probs'].values[i])]
            # generate the dataframe
            DF_TEST = pd.DataFrame({
                # Model configuration
                "i" : df.iloc[i][7],
                "o" : df.iloc[i][8],
                "io" : df.iloc[i][1],
                "technique" : df.iloc[i][2],
                "Session" : df.iloc[i][25],
                "Trial" : df.iloc[i][4],
                "path" : df.iloc[i][24],
                "n_samples" : df.iloc[i][5],
                "n_samples_cc" : df.iloc[i][6],
                "i_is_conf" : df.iloc[i][9],
                "o_is_conf" : df.iloc[i][10],
                "Model" : df.iloc[i][3],
                "model_SVM-rbf__C" : df.iloc[i][18],
                "model_SVM-rbf__gamma" : df.iloc[i][19],
                "runtime" : df.iloc[i][20],
                "model_SVM-lin__C" : df.iloc[i][21],
                "model_GB__learning_rate" : df.iloc[i][22],
                "model_LR__C" : df.iloc[i][23],
                # Result
                "train_score" : df.iloc[i][11],
                "valid_score" : df.iloc[i][12],
                "test_score" : df.iloc[i][13],
                "roc_auc" : df.iloc[i][14],
                "holdout_score" : df.iloc[i][26],
                "holdout_roc_auc" : df.iloc[i][27],
                # Test
                "dataset" : "Test set",
                "ID" : test_ids,
                "true_label" : test_lbls,
                "prediction" : test_probs,
            })
            DF_HOLDOUT = pd.DataFrame({
                # Model configuration
                "i" : df.iloc[i][7],
                "o" : df.iloc[i][8],
                "io" : df.iloc[i][1],
                "technique" : df.iloc[i][2],
                "Session" : df.iloc[i][25],
                "Trial" : df.iloc[i][4],
                "path" : df.iloc[i][24],
                "n_samples" : df.iloc[i][5],
                "n_samples_cc" : df.iloc[i][6],
                "i_is_conf" : df.iloc[i][9],
                "o_is_conf" : df.iloc[i][10],
                "Model" : df.iloc[i][3],
                "model_SVM-rbf__C" : df.iloc[i][18],
                "model_SVM-rbf__gamma" : df.iloc[i][19],
                "runtime" : df.iloc[i][20],
                "model_SVM-lin__C" : df.iloc[i][21],
                "model_GB__learning_rate" : df.iloc[i][22],
                "model_LR__C" : df.iloc[i][23],
                # Result
                "train_score" : df.iloc[i][11],
                "valid_score" : df.iloc[i][12],
                "test_score" : df.iloc[i][13],
                "roc_auc" : df.iloc[i][14],
                "holdout_score" : df.iloc[i][26],
                "holdout_roc_auc" : df.iloc[i][27],
                # Holdout
                "dataset" : "Holdout set",
                "ID" : holdout_ids,
                "true_label" : holdout_lbls,
                "prediction" : holdout_preds
            })
            DF.append(DF_TEST)
            DF.append(DF_HOLDOUT)
            
        # generate the columns
        DF2 = pd.concat(DF).reset_index(drop=True)
        
        TP = (DF2.true_label == 1.0) & (DF2.prediction >= 0.5)
        TN = (DF2.true_label == 0.0) & (DF2.prediction < 0.5)
        FP = (DF2.true_label == 0.0) & (DF2.prediction >= 0.5)
        FN = (DF2.true_label == 1.0) & (DF2.prediction < 0.5)
        
        DF2['TP prob'] = DF2[TP]['prediction']
        DF2['TN prob'] = DF2[TN]['prediction']
        DF2['FP prob'] = DF2[FP]['prediction']
        DF2['FN prob'] = DF2[FN]['prediction']
        DF2['T prob'] = DF2[TP | TN]['prediction']
        DF2['F prob'] = DF2[FP | FN]['prediction']
        
        conditionlist = [
            (DF2['TP prob'] >= 0.5) ,
            (DF2['TN prob'] < 0.5) ,
            (DF2['FP prob'] >= 0.5),
            (DF2['FN prob'] < 0.5)]
        choicelist = ['TP', 'TN', 'FP', 'FN']
        DF2['Prob'] = np.select(conditionlist, choicelist,
                                default='Not Specified')
        
        conditionlist2 = [
            (DF2['Prob'] == 'TP') | (DF2['Prob'] == 'TN'),
            (DF2['Prob'] == 'FP') | (DF2['Prob'] == 'FN')]
        choicelist2 = ['TP & TN', 'FP & FN']
        DF2['Predict TF'] = np.select(conditionlist2, choicelist2,
                                      default='Not Specified')
        
        conditionlist3 = [
            (DF2['Prob'] == 'TP') | (DF2['Prob'] == 'FP'),
            (DF2['Prob'] == 'TN') | (DF2['Prob'] == 'FN')]
        choicelist3 = ['TP & FP', 'TN & FN']
        DF2['Model PN'] = np.select(conditionlist3, choicelist3,
                                    default='Not Specified')

        conditionlist4 = [
            (DF2['Prob'] == 'TP') | (DF2['Prob'] == 'FN'),
            (DF2['Prob'] == 'TN') | (DF2['Prob'] == 'FP')]
        choicelist4 = ['TP & FN', 'TN & FP']
        DF2['Label PN'] = np.select(conditionlist4, choicelist4,
                                    default='Not Specified')
        # rename the values may be needed
        DF2['Session'] = DF2['Session'].map({'bl':'BL',
                                             'fu1':'FU1',
                                             'fu2':'FU2',
                                             'fu3':'FU3'})
        if save == True:
            save_path = f"{self.DATA_DIR}/posthoc/all_RUN.csv"
            if not os.path.isdir(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            DF2.to_csv(save_path, index=None)
        return DF2

    def get_RUN(self, RUN_file):
        """ Load the RUN file
        
        Parameters
        ----------            
        RUN_file : string
            The IMAGEN's RUN file (*.csv)

        Returns
        -------
        DF : pandas.dataframe
            The RUN dataframe
        
        Notes
        -----
        This function select the RUN
        
        Examples
        --------
        >>> from imagen_posthocloader import *
        >>> DATA = IMAGEN_instrument()
        >>> DF = DATA.get_RUN(
        ...     RUN_file)                             # RUN
        >>> DF_FU3 = DF.groupby('Session').get_group('FU3')

        """
        # Load the instrument file       
        run_path = f"{self.DATA_DIR}/posthoc/{RUN_file}"
        DF = pd.read_csv(run_path, low_memory=False)
        return DF
    
#     def __str__(self):
#         pass

class SHAP_loader:
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
        >>> from imagen_posthocloader import *
        >>> MODEL = SHAP_loader()
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
            models.update({model_name: [load(f) for f in glob(models_dir+f"/{model_name}_cb_*.model")]})
        self.MODELS = models
        self.MODEL_NAME = model_names
        return self.MODELS
    
    def get_list(self, MODELS, X, md='All'):
        """ Generate the SHAP input value list
        
        Parameters
        ----------
        MODELS : dictionary
            model configuration
        X : numpy.ndarray
            Data, hdf5 file
        md : string
            Specify the model: SVM-RBF, SVM-LIN, LR, GB
        Returns
        -------
        self.INPUT : list
            SHAP input combination list
        
        Examples
        --------
        >>> from imagen_posthocloader import *
        >>> DATA = SHAP_loader()
        >>> INPUT = DATA.get_list(
        ...     'MODELS',                 # MODELS
        ...     'X',                      # X
        ...     md='All')                 # model
        
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
        if md =='All':
            self.INPUT = INPUT
        else:
            self.INPUT = INPUT = [i for i in INPUT if i[0]==md]
        return self.INPUT
    
    def get_SHAP(self, INPUT, NAME, save = True):
        """ Generate the SHAP value
        
        Parameters
        ----------
        INPUT: list
            SHAP INPUT: Model name, X, and N - trial number
        save : boolean
            Defualt save the shap_value
        
        Examples
        --------
        >>> from imagen_posthocloader import *
        >>> DATA = SHAP_loader()
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
        X100 = shap.utils.sample(X, len(X))
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
                    if not os.path.isdir("test"):
                        os.makedirs("test")
                    with open(f"test/{model_name+str(i)}_{NAME}.sav", "wb") as f:
                        pickle.dump(shap_values, f)

    def load_abs_SHAP(self, SHAP):
        """ Generate the mean and std of|SHAP value| 
        
        Parameters
        ----------
        SHAP : .sav file
            Load the SHAP value
        
        Examples
        --------
        >>> from imagen_posthocloader import *
        >>> DATA = SHAP_loader()
        >>> mean_SHAP, std_SHAP = DATA.load_SHAP(
        ...     'SHAP')                     # SHAP
        
        """
        with open(self.DATA_DIR+"/posthoc/explainers/"+SHAP, 'rb') as fp:
            load_shap_values = pickle.load(fp)
        SHAP_list = []
        for data in load_shap_values:
            value = [data[i].values for i in range(data.shape[0])]
            SHAP_list.append(value)
        DF_SHAP = pd.DataFrame(SHAP_list)
        mean_SHAP = list(DF_SHAP.apply(abs).mean())
        std_SHAP = list(DF_SHAP.apply(abs).std())
        return mean_SHAP, std_SHAP
    
    def load_SHAP(self, HDF5, SHAP, save=False):
        """ Generate the pandas dataframe of the SHAP value
        
        Parameters
        ----------
        HDF5 : .hdf5 file
            Load the ID, X_Col_names value
        
        SHAP : .sav file
            Load the SHAP value
        
        Returns
        -------
        DF : pandas.dataframe
            SHAP value dataframe
        
        Note
        ----
        Sex mask is not implemented
        
        Examples
        --------
        >>> from imagen_posthocloader import *
        >>> DATA = SHAP_loader()
        >>> mean_SHAP, std_SHAP = DATA.load_SHAP(
        ...     'HDF5',                      # HDF5
        ...     'SHAP')                      # SHAP    
        
        """
        _, X_col, Other = self.get_holdout_data(HDF5)
        with open(self.DATA_DIR+"/posthoc/explainers/"+SHAP, 'rb') as fp:
            load_shap_values = pickle.load(fp)
        
        Info = SHAP.split('_')
        Model = Info[0][:-1]
        Session = Info[-1].replace(".sav","")
        Trial = Info[0][-1:]
        Class = ['HC' if i==0 else 'AAM' for i in Other[0]]
        ID = Other[1]
        
        df = pd.DataFrame(
            {'ID' : ID,
             'Session' : Session,
             'Trial' : Trial,
             'Model' : Model,
             'Class' : Class
            }
        )
        df2 = pd.DataFrame(load_shap_values.values, columns=X_col)
        DF = pd.concat([df, df2], axis=1)
        
        if save == True:
            save_path = f"{self.DATA_DIR}/posthoc/explainers/all_{Info[0]}_{Session}_SHAP.csv"
            # set the save option
            if not os.path.isdir(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            DF.to_csv(save_path, index=None)
        return DF

    def load_Feature(self, HDF5, SHAP):
        """ Generate the pandas dataframe of the Feature value
        
        Parameters
        ----------
        HDF5 : .hdf5 file
            Load the X_Col_names value
            
        SHAP : .sav file
            Load the SHAP value
        
        Returns
        -------
        DF : pandas.dataframe
            Feature values, only SHAP values > 0 above, dataframe
        
        Examples
        --------
        >>> from imagen_posthocloader import *
        >>> DATA = SHAP_loader()
        >>> mean_SHAP, std_SHAP = DATA.load_POS_Feature(
        ...     SHAP                      # SHAP    
        ...     save=False)               # Save
        
        """
        # load the X_col_name
        _, X_col, _ = self.get_holdout_data(HDF5)
        
        # load the data
        SHAP = f"{self.DATA_DIR}/posthoc/explainers/{SHAP}"
        with open(SHAP, 'rb') as fp:
            load_shap_values = pickle.load(fp)
            
        data = load_shap_values.data
        
        # load the standardized values
        scaler = StandardScaler()
        scaler.fit(data)
        new_data = scaler.transform(data)
        df = pd.DataFrame(new_data, columns=X_col)
        display(df.describe())
        
        # load the SHAP values
        new_shap = load_shap_values.values
        df2 = pd.DataFrame(new_shap, columns=X_col)
        df3 = np.sign(df2)
        df4 = pd.DataFrame()
        for i in df3.columns:
            df4[i] = df3[i].apply(lambda x: 1)
        
        # generate the feature values
        DF = df4*df
        
        return DF
    
    def load_POS_Feature(self, HDF5, SHAP):
        """ Generate the pandas dataframe of the Feature value which above 0 of the SHAP vlaues
        
        Parameters
        ----------
        HDF5 : .hdf5 file
            Load the X_Col_names value
            
        SHAP : .sav file
            Load the SHAP value
        
        Returns
        -------
        DF : pandas.dataframe
            Feature values, only SHAP values > 0 above, dataframe
        
        Examples
        --------
        >>> from imagen_posthocloader import *
        >>> DATA = SHAP_loader()
        >>> mean_SHAP, std_SHAP = DATA.load_POS_Feature(
        ...     SHAP                      # SHAP    
        ...     save=False)               # Save
        
        """
        # load the X_col_name
        _, X_col, _ = self.get_holdout_data(HDF5)
        
        # load the data
        SHAP = f"{self.DATA_DIR}/posthoc/explainers/{SHAP}"
        with open(SHAP, 'rb') as fp:
            load_shap_values = pickle.load(fp)
            
        data = load_shap_values.data
        
        # load the standardized values
        scaler = StandardScaler()
        scaler.fit(data)
        new_data = scaler.transform(data)
        df = pd.DataFrame(new_data, columns=X_col)
#         display(df.describe())
        
        # load the SHAP values
        new_shap = load_shap_values.values
        df2 = pd.DataFrame(new_shap, columns=X_col)
        df3 = np.sign(df2)
        df4 = pd.DataFrame()
        for i in df3.columns:
            df4[i] = df3[i].apply(lambda x: 1 if x > 0 else np.NaN)
        
        # generate the feature values
        DF = df4*df
        
        return DF

    def check_POS_Feature(self, HDF5, SHAP):
        """ Generate the pandas dataframe of the Feature value which above 0 of the SHAP vlaues
        
        Parameters
        ----------
        HDF5 : .hdf5 file
            Load the X_Col_names value
            
        SHAP : .sav file
            Load the SHAP value
        
        Returns
        -------
        DF : pandas.dataframe
            Feature values, only SHAP values > 0 above, dataframe
        
        Examples
        --------
        >>> from imagen_posthocloader import *
        >>> DATA = SHAP_loader()
        >>> mean_SHAP, std_SHAP = DATA.load_POS_Feature(
        ...     SHAP                      # SHAP    
        ...     save=False)               # Save
        
        """
        # load the X_col_name
        _, X_col, _ = self.get_holdout_data(HDF5)
        
        # load the data
        SHAP = f"{self.DATA_DIR}/posthoc/explainers/{SHAP}"
        with open(SHAP, 'rb') as fp:
            load_shap_values = pickle.load(fp)
            
        data = load_shap_values.data
        
        # load the standardized values
        scaler = StandardScaler()
        scaler.fit(data)
        new_data = scaler.transform(data)
        df = pd.DataFrame(new_data, columns=X_col)
        display(df.describe())
        
        # load the SHAP values
        new_shap = load_shap_values.values
        df2 = pd.DataFrame(new_shap, columns=X_col)
#         df3 = np.sign(df2)
#         df4 = pd.DataFrame()
#         for i in df2.columns:
#             df4[i] = df2[i].apply(lambda x: 1 if x > 0 else 0)
        
        return df2

    def load_NEG_Feature(self, HDF5, SHAP):
        """ Generate the pandas dataframe of the Feature value which below 0 of the SHAP vlaues
        
        Parameters
        ----------
        HDF5 : .hdf5 file
            Load the X_Col_names value
            
        SHAP : .sav file
            Load the SHAP value
        
        Returns
        -------
        DF : pandas.dataframe
            Feature values, only SHAP values < 0 , dataframe
        
        Examples
        --------
        >>> from imagen_posthocloader import *
        >>> DATA = SHAP_loader()
        >>> mean_SHAP, std_SHAP = DATA.load_POS_Feature(
        ...     SHAP                      # SHAP    
        ...     save=False)               # Save
        
        """
        # load the X_col_name
        _, X_col, _ = self.get_holdout_data(HDF5)
        
        # load the data
        SHAP = f"{self.DATA_DIR}/posthoc/explainers/{SHAP}"
        with open(SHAP, 'rb') as fp:
            load_shap_values = pickle.load(fp)
            
        data = load_shap_values.data
        
        # load the standardized values
        scaler = StandardScaler()
        scaler.fit(data)
        new_data = scaler.transform(data)
        df = pd.DataFrame(new_data, columns=X_col)
        
        # load the SHAP values
        new_shap = load_shap_values.values
        df2 = pd.DataFrame(new_shap, columns=X_col)
        df3 = np.sign(df2)
        df4 = pd.DataFrame()
        for i in df3.columns:
            df4[i] = df3[i].apply(lambda x: 1 if x < 0 else np.NaN)
        
        # generate the feature values
        DF = df4*df
        
        return DF
    
    def load_ZER_Feature(self, HDF5, SHAP):
        """ Generate the pandas dataframe of the Feature value which is 0 of the SHAP vlaues
        
        Parameters
        ----------
        HDF5 : .hdf5 file
            Load the X_Col_names value
            
        SHAP : .sav file
            Load the SHAP value
        
        Returns
        -------
        DF : pandas.dataframe
            Feature values, only SHAP values = 0 , dataframe
        
        Examples
        --------
        >>> from imagen_posthocloader import *
        >>> DATA = SHAP_loader()
        >>> mean_SHAP, std_SHAP = DATA.load_POS_Feature(
        ...     SHAP                      # SHAP    
        ...     save=False)               # Save
        
        """
        # load the X_col_name
        _, X_col, _ = self.get_holdout_data(HDF5)
        
        # load the data
        SHAP = f"{self.DATA_DIR}/posthoc/explainers/{SHAP}"
        with open(SHAP, 'rb') as fp:
            load_shap_values = pickle.load(fp)
            
        data = load_shap_values.data
        
        # load the standardized values
        scaler = StandardScaler()
        scaler.fit(data)
        new_data = scaler.transform(data)
        df = pd.DataFrame(new_data, columns=X_col)
        
        # load the SHAP values
        new_shap = load_shap_values.values
        df2 = pd.DataFrame(new_shap, columns=X_col)
        df3 = np.sign(df2)
        df4 = pd.DataFrame()
        for i in df3.columns:
            df4[i] = df3[i].apply(lambda x: 1 if x == 0 else np.NaN)
        
        # generate the feature values
        DF = df4*df
        
        return DF

#################################################################################
#                                 POST HOC ANALYSIS                             #
#################################################################################
def load_all_labels(x_tp="FU3", val_type='binarize', exclude_holdout=False): #, "FU2", "BL"
    x_tp = x_tp.upper()
    assert x_tp in ["FU3", "FU2", "BL"]
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

        # ----------------------------------------------------- #
        # Summary Statistics Plot                               #
        # ----------------------------------------------------- #   
# (Todo) Convert into Class format
INSTRUMENTS_DIR = '/ritter/share/data/IMAGEN/posthoc/'
def show_corr_with_instrument(instrument, session, 
                              use_only_lbls=[], ignore_cols=['Sex', 'Site', 'Session'],
                              val_type='categorize', corr_type='pearsonr'):
    
    # 1) Load the instrument
    if isinstance(instrument, pd.DataFrame):
        instrument_df = instrument.set_index('ID')
    else:
        instrument_df = pd.read_csv(f"{INSTRUMENTS_DIR}/all_{instrument}.csv").set_index('ID')
    # select FU3 if available or else FU2 or else BL
    if 'FU3' in instrument_df['Session'].unique():
        instrument_df = instrument_df.groupby('Session').get_group('FU3')
    elif 'FU2' in instrument_df['Session'].unique():
        instrument_df = instrument_df.groupby('Session').get_group('FU2')
    else:
        instrument_df = instrument_df.groupby('Session').get_group('BL')
    
    # remove the suffix 'mean' or 'sum' from the column names for better presentation
    instrument_df = instrument_df.rename(lambda c: c.replace(' mean', '').replace('Sexsual', 'Sexual'), axis='columns')
    instrument_cols = [c for c in instrument_df.columns if c.title() not in ignore_cols]
    instrument_df = instrument_df.reindex(instrument_cols, axis='columns')

    # if the values are categorical convert them to ints
    for c in instrument_cols:
        if isinstance(instrument_df[c].dropna().unique()[0], str): 
            instrument_df[c] = pd.Categorical(instrument_df[c]).codes

    # 2) Load the AAM phenotypes
    df_all_labels = load_all_labels(session, val_type=val_type, exclude_holdout=False)
    label_cols = [c for c in df_all_labels.columns if c.title() not in ignore_cols]
    # select only the request AAM labels
    if use_only_lbls:
        label_cols = use_only_lbls
        # drop others and also reorder the columns as requested
        df_all_labels = df_all_labels.reindex(use_only_lbls, axis='columns')
        
    # 3) merge 1 and 2
    df = df_all_labels.join(instrument_df).copy()
    
    # 4) compute the correlation and return the pandas Styler object
    corr, pvals = get_corr(df, corr_type, cols=label_cols, rows=instrument_cols)
    
    display_html(corr._repr_html_() + pvals._repr_html_(), raw=True)

        # ----------------------------------------------------- #
        # Sensitivity Analysis Plot                             #
        # ----------------------------------------------------- #   
# plot the correlations between the different models
def get_corr(df, corr_type="r-squared", cols=None, rows=None, vmax=None, vmin=None, mask_diag_repeatitions=False):
    
    def r_squared(a, b): 
        if isinstance(a.iloc[0], str): a = pd.Categorical(a).codes
        if isinstance(b.iloc[0], str): b = pd.Categorical(b).codes
        return (np.corrcoef(a, b)[0,1]**2)*100, np.NaN
    
    def accuracy(a, b): return accuracy_score(a, b)*100, np.NaN # no p_val calulated
    
    def chi_squared(a, b): return stats.chi2_contingency(pd.crosstab(a, b))[:2]
    
    # calculate the correlations
    corr_type = corr_type.lower()
    if corr_type in ['r', 'correlation', 'pearsonr', 'kendalltau', 'spearmanr']:
        if vmax is None: vmax=0.5
        if vmin is None: vmin=-0.5
        cmap = 'RdBu'
        try:
            corr_type = getattr(stats, corr_type)
        except: # default to pearsonr correlation
            corr_type = stats.pearsonr
        fmt = '{:.2f}'
    elif corr_type in ['r2','r_squared','r-squared']:
        if vmax is None: vmax=90
        if vmin is None: vmin=0
        cmap = 'PuBu'
        corr_type = r_squared
        fmt = '{:.1f}%'
    elif corr_type in ['agreement', "accuracy"]:
        if vmax is None: vmax=0.95
        if vmin is None: vmin=0
        cmap = 'YlGnBu'
        corr_type = accuracy
        fmt = '{:.0f}%'
    elif corr_type in ['chi2', 'chi-squared']:
        if vmax is None: vmax=500
        if vmin is None: vmin=0
        cmap = 'YlGnBu'
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
            
    if mask_diag_repeatitions:
            # Fill diagonal and upper half with NaNs
            mask = np.zeros_like(df_corr, dtype=bool)
            mask[np.triu_indices_from(mask)] = True
            df_corr[mask] = np.nan
          
    corr = df_corr.apply(pd.to_numeric).style.format(fmt).background_gradient(
                vmin=vmin, vmax=vmax, cmap=cmap).highlight_null( # make NaNs grey
        null_color='#f1f1f1').set_table_attributes("style='display:inline'").set_caption(corr_type.__name__.title())
    
    # also color style the p_vals
    def p_val_colors(v):
        cmap = plt.get_cmap('Reds')
        if v <= 0.001:
            return f"background-color: {mpl.colors.rgb2hex(cmap(0.8))};"  
        elif v <= 0.01:
            return f"background-color: {mpl.colors.rgb2hex(cmap(0.6))};"  
        elif v <= 0.05:
            return f"background-color: {mpl.colors.rgb2hex(cmap(0.5))};"  
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

def plot_subject_classes_modelwise(model_results, use_probs, 
                                   lbl='Binge', models=["SVM-rbf","GB","SVM-lin","LR"],
                                  only_corr=False):
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
        
        if not only_corr:
            # visualize the subject class between different models
            sort_order = [lbl,*models]

            fig, ax = plt.subplots(figsize=(16,len(sort_order)))

            plot_subject_classes(df_ml,
                                 ax, confs=['Sex', 'Site'],
                                 sort_order=sort_order, 
                                 title='Comparison of Model predictions on the same subject',
                                 lw=30, cmap=plt.cm.YlGnBu)
            plt.show()


        
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

### helper funcs
def get_featuretype_cnt(fs):
    dfc = pd.DataFrame()
    dfc.loc[0, 'Total'] = len(fs)
    dfc.loc[0, 'DTI'] = len([f for f in fs if 'DTI'==f.split('_')[0]])
    dfc.loc[0, 'T1w'] = len([f for f in fs if 'T1w'==f.split('_')[0]])
    dfc.loc[0, 'subcor'] = len([f for f in fs if 'subcor'==f.split('_')[1]])
    dfc.loc[0, 'subcor_area'] = len([f for f in fs if 'subcor'==f.split('_')[1] and 'mean'==f.split('_')[-1]])
    dfc.loc[0, 'subcor_vol'] = len([f for f in fs if 'subcor'==f.split('_')[1] and 'volume'==f.split('_')[-1]])
    
    dfc.loc[0, 'cor'] = len([f for f in fs if 'cor'==f.split('_')[1]])
    dfc.loc[0, 'cor_area'] = len([f for f in fs if 'cor'==f.split('_')[1] and 'area'==f.split('-')[-1]])
    dfc.loc[0, 'cor_curv'] = len([f for f in fs if 'cor'==f.split('_')[1] and 'curv' in f.split('-')[-1]])
    dfc.loc[0, 'cor_vol'] = len([f for f in fs if 'cor'==f.split('_')[1] and 'vol' in f.split('-')[-1]])
    dfc.loc[0, 'cor_thick'] = len([f for f in fs if 'cor'==f.split('_')[1] and 'thickness' in f.split('-')[-1]])
    dfc.loc[0, 'cor_foldind'] = len([f for f in fs if 'cor'==f.split('_')[1] and 'foldind' == f.split('-')[-1]])
    
    dfc = dfc.astype(int)
    
    return dfc.style.background_gradient(cmap='gray', vmin=0, vmax=len(fs))

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

#################################################################################
#                              MODEL INTERPRETATION                             #
#################################################################################
            
def SHAP_table(DF, SESSION, viz = False):
    # DTI type
    DTI0 = [i for i in zip(DF[f'sorted SVM-rbf0_{SESSION} name'], DF[f'sorted SVM-rbf0_{SESSION} mean'], DF[f'sorted SVM-rbf0_{SESSION} std']) if 'DTI_' in i[0]]
    DTI1 = [i for i in zip(DF[f'sorted SVM-rbf1_{SESSION} name'], DF[f'sorted SVM-rbf1_{SESSION} mean'], DF[f'sorted SVM-rbf1_{SESSION} std']) if 'DTI_' in i[0]]
    DTI2 = [i for i in zip(DF[f'sorted SVM-rbf2_{SESSION} name'], DF[f'sorted SVM-rbf2_{SESSION} mean'], DF[f'sorted SVM-rbf2_{SESSION} std']) if 'DTI_' in i[0]]
    DTI3 = [i for i in zip(DF[f'sorted SVM-rbf3_{SESSION} name'], DF[f'sorted SVM-rbf3_{SESSION} mean'], DF[f'sorted SVM-rbf3_{SESSION} std']) if 'DTI_' in i[0]]
    DTI4 = [i for i in zip(DF[f'sorted SVM-rbf4_{SESSION} name'], DF[f'sorted SVM-rbf4_{SESSION} mean'], DF[f'sorted SVM-rbf4_{SESSION} std']) if 'DTI_' in i[0]]
    DTI5 = [i for i in zip(DF[f'sorted SVM-rbf5_{SESSION} name'], DF[f'sorted SVM-rbf5_{SESSION} mean'], DF[f'sorted SVM-rbf5_{SESSION} std']) if 'DTI_' in i[0]]
    DTI6 = [i for i in zip(DF[f'sorted SVM-rbf6_{SESSION} name'], DF[f'sorted SVM-rbf6_{SESSION} mean'], DF[f'sorted SVM-rbf6_{SESSION} std']) if 'DTI_' in i[0]]
    # T1w Subcortical type
    SUBCOR0 = [i for i in zip(DF[f'sorted SVM-rbf0_{SESSION} name'], DF[f'sorted SVM-rbf0_{SESSION} mean'], DF[f'sorted SVM-rbf0_{SESSION} std']) if 'T1w_subcor_' in i[0]]
    SUBCOR1 = [i for i in zip(DF[f'sorted SVM-rbf1_{SESSION} name'], DF[f'sorted SVM-rbf1_{SESSION} mean'], DF[f'sorted SVM-rbf1_{SESSION} std']) if 'T1w_subcor_' in i[0]]
    SUBCOR2 = [i for i in zip(DF[f'sorted SVM-rbf2_{SESSION} name'], DF[f'sorted SVM-rbf2_{SESSION} mean'], DF[f'sorted SVM-rbf2_{SESSION} std']) if 'T1w_subcor_' in i[0]]
    SUBCOR3 = [i for i in zip(DF[f'sorted SVM-rbf3_{SESSION} name'], DF[f'sorted SVM-rbf3_{SESSION} mean'], DF[f'sorted SVM-rbf3_{SESSION} std']) if 'T1w_subcor_' in i[0]]
    SUBCOR4 = [i for i in zip(DF[f'sorted SVM-rbf4_{SESSION} name'], DF[f'sorted SVM-rbf4_{SESSION} mean'], DF[f'sorted SVM-rbf4_{SESSION} std']) if 'T1w_subcor_' in i[0]]
    SUBCOR5 = [i for i in zip(DF[f'sorted SVM-rbf5_{SESSION} name'], DF[f'sorted SVM-rbf5_{SESSION} mean'], DF[f'sorted SVM-rbf5_{SESSION} std']) if 'T1w_subcor_' in i[0]]
    SUBCOR6 = [i for i in zip(DF[f'sorted SVM-rbf6_{SESSION} name'], DF[f'sorted SVM-rbf6_{SESSION} mean'], DF[f'sorted SVM-rbf6_{SESSION} std']) if 'T1w_subcor_' in i[0]]
    # T2w Subcortical type
    COR0 = [i for i in zip(DF[f'sorted SVM-rbf0_{SESSION} name'], DF[f'sorted SVM-rbf0_{SESSION} mean'], DF[f'sorted SVM-rbf0_{SESSION} std']) if 'T1w_cor_' in i[0]]
    COR1 = [i for i in zip(DF[f'sorted SVM-rbf1_{SESSION} name'], DF[f'sorted SVM-rbf1_{SESSION} mean'], DF[f'sorted SVM-rbf1_{SESSION} std']) if 'T1w_cor_' in i[0]]
    COR2 = [i for i in zip(DF[f'sorted SVM-rbf2_{SESSION} name'], DF[f'sorted SVM-rbf2_{SESSION} mean'], DF[f'sorted SVM-rbf2_{SESSION} std']) if 'T1w_cor_' in i[0]]
    COR3 = [i for i in zip(DF[f'sorted SVM-rbf3_{SESSION} name'], DF[f'sorted SVM-rbf3_{SESSION} mean'], DF[f'sorted SVM-rbf3_{SESSION} std']) if 'T1w_cor_' in i[0]]
    COR4 = [i for i in zip(DF[f'sorted SVM-rbf4_{SESSION} name'], DF[f'sorted SVM-rbf4_{SESSION} mean'], DF[f'sorted SVM-rbf4_{SESSION} std']) if 'T1w_cor_' in i[0]]
    COR5 = [i for i in zip(DF[f'sorted SVM-rbf5_{SESSION} name'], DF[f'sorted SVM-rbf5_{SESSION} mean'], DF[f'sorted SVM-rbf5_{SESSION} std']) if 'T1w_cor_' in i[0]]
    COR6 = [i for i in zip(DF[f'sorted SVM-rbf6_{SESSION} name'], DF[f'sorted SVM-rbf6_{SESSION} mean'], DF[f'sorted SVM-rbf6_{SESSION} std']) if 'T1w_cor_' in i[0]]
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

def SHAP_plot(DATA, SHAP, TYPE, fig=False):
    X, X_col_names, Other_list = IMAGEN_posthoc().get_holdout_data(DATA, group=True)
    with open(SHAP, 'rb') as fp:
        load_shap_values = pickle.load(fp)

    ROI = SHAP.split('/')[-1].replace('.sav','')
    if TYPE == 'Bar':
        shap.summary_plot(load_shap_values, features=X, feature_names=X_col_names, plot_type='bar', show=False)
        plt.title(f"{ROI} Bar plot")
    
    if TYPE == 'Swarm':
        shap.summary_plot(load_shap_values, features=X, feature_names=X_col_names, plot_type='dot', show=False)
        plt.title(f"{ROI} Swarm plot")
        
    if TYPE == 'Class':
        sex_mask = Other_list[3].astype(bool)
        shap.group_difference_plot(load_shap_values.values, group_mask=sex_mask,
                                   feature_names=X_col_names, show=False, max_display=10)
        plt.title(f"{ROI} Class group difference plot")
        
    if TYPE == 'Sex':
        sex_mask = Other_list[2].astype(bool)
        shap.group_difference_plot(load_shap_values.values, group_mask=sex_mask,
                                   feature_names=X_col_names, show=False, max_display=10)
        plt.title(f"{ROI} Sex group difference plot")
    
    if fig == True:
        if not os.path.isdir('figures'):
            os.makedirs('figures')
        plt.savefig(f"figures/{ROI}_{TYPE}_plot.pdf", bbox_inches='tight')

class IMAGEN_posthoc(INSTRUMENT_loader,HDF5_loader,RUN_loader,SHAP_loader):
    def __init__(self, DATA_DIR="/ritter/share/data/IMAGEN"):
        """ Set up path
        
        Parameters
        ----------
        DATA_DIR : string, optional
            Directory IMAGEN absolute path
        
        """
        # Set the directory path: IMAGEN
        self.DATA_DIR = DATA_DIR
        
    def to_INSTRUMENT(self, LIST, save=False):
        """ Merge the ROI instruments as one file
        
        Parameters
        ----------
        LIST : list
            instrument roi name list
        save : boolean
            save it in the folder
            
        Returns
        -------
        self.INSTRUMENT : pandas.dataframe
            The selected INSTRUMENT dataframe
            
        Notes
        -----
        Each Instrument has different Session and ID cases
        
        Examples
        --------
        >>> from imagen_posthocloader import *
        >>> DATA = IMAGEN_posthoc()
        >>> DF3 = DATA.to_INSTURMENT(
        ...     LIST,                            # instrument list
        ...     save = True)

        """
        if len(LIST)==1:
            self.INSTRUMENT = LIST[0]
        else:
            Z = pd.merge(LIST[0],LIST[1],on=['ID','Session'],how='outer')
            if len(LIST)!=2:
                for n in LIST[2:]:
                    Z = pd.merge(Z,n,on=['ID','Session'],how='outer')
        self.INSTRUMENT = Z

        if save == True:
            save_path = f"{self.DATA_DIR}/posthoc/IMAGEN_INSTRUMENT.csv"
            # set the save option
            if not os.path.isdir(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            Z.to_csv(save_path, index=None)
        return self.INSTRUMENT

    def to_HDF5(self, hdf5_file, save=False):
        """ Generate the dataframe,
        all subject (ALL), healthy control (HC),
        adolscent alcohol misuse (AAM), Sex, Site, and Class
        
        Parameters
        ----------
        hdf5_file : string,
            The IMAGEN's h5df file (*.csv)
        save : boolean
            if save == True, then save it as .csv
        
        Returns
        -------
        self.HDF5 : pandas.dataframe
            The slected HDF5 dataframe
            
        Examples
        --------
        >>> from imagen_posthocloader import *
        >>> DATA = IMAGEN_posthoc()
        >>> HDF5 = DATA.to_HDF5(
        ...     h5py_file,                               # HDF5
        ...     save = True)                             # Save
        >>> HDF5_FU3 = HDF5.groupby('Session').get_group('fu3')

        Notes
        -----
        Dataset:
        Training and Holdout

        """
        # Load the hdf5 file
        DF = self.get_HDF5(hdf5_file)
        self.HDF5 = DF
        if save == True:
            save_path = f"{self.DATA_DIR}/posthoc/IMAGEN_HDF5.csv"
            if not os.path.isdir(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            DF.to_csv(save_path, index=None)
        return self.HDF5    
    
    def to_RUN(self, run_file, COL, save=False):
        """ Select the ROI columns in one file
        
        Parameters
        ----------
        run_file : string
            ML models result run.csv path
        COL : list
            ROI columns
        save : boolean
            if save == True, then save it as .csv
        
        Returns
        -------
        self.RUN : pandas.dataframe
            The RUN dataframe
        
        Examples
        --------
        >>> from imagen_posthocloader import *
        >>> DATA = IMAGEN_posthoc()
        >>> RUN = DATA.to_RUN(
        ...     run_file)                               # run
        >>> RUN_FU3 = RUN.groupby('Session').get_group('fu3')
        
        Notes
        -----
        There are more options to select ROI columns
        General information:
            'ID','Session','Trial','dataset','io','technique','Model',
            'TP prob','TN prob','FP prob','FN prob','T prob','F prob','Prob',
            'Predict TF','Model PN','Label PN','true_label','prediction'
        
        It may extend to other y cases in one file
        
        """
        DF = self.get_RUN(run_file)
        DF2 = DF[COL]

        self.RUN = DF2
        if save == True:
            save_path = f"{self.DATA_DIR}/posthoc/IMAGEN_RUN.csv"
            if not os.path.isdir(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            DF2.to_csv(save_path, index=None)
        return self.RUN
    
    def to_abs_SHAP(self, H5, SHAP, SESSION, save=False):
        """ Get the subtype and mean and std of |SHAP value| in given models
            Get the std, and mean of mean|SHAP value| in given models
        
        Parameters
        ----------
        H5: string
            Feature name contained H5 list
        SHAP : list
            Model list of |SHAP value|
        save : boolean
            if save == True, then save it as .csv
        
        Returns
        -------
        COL : pandas.dataframe
            Region category of the Feature
            The mean, std |SHAP value|
            The mean, std of mean |SHAP value|

        Examples
        --------
        >>> from imagen_posthocloader import *
        >>> DATA = IMAGEN_posthoc()
        >>> RUN = DATA.to_abs_SHAP(
        ...     H5,                             # HDF5 for Feature name
        ...     SHAP,                           # list of SHAP model name
        ...     save=False)                     # save
        """
        
        def type_check(col):
            """ Generate the region category of the Feature
            
            Parameters
            ----------
            col: string
                Feature name
                
            Returns
            -------
            * region : String
                Return the * region under the condition
            
            """
            if 'cor' == col.split('_')[1]:
                return "Cortical region"
            elif 'subcor' == col.split('_')[1]:
                return "Subcortical region"
            else:
                return "DTI region"
    
        def lobe_region(col):
            """ Generate the lobe region of the Feature
            
            Parameters
            ----------
            col: string
                Feature name
                
            Returns
            -------
            * lobe or * cortex or * region : string
                Return the lobe region name
                
            """
            if 'cor' == col.split('_')[1]:
                temporal_lobe = {'bankssts', 'entorhinal', 'fusiform', 'inferiortemporal', 'middletemporal',
                                 'parahippocampal','superiortemporal', 'temporalpole', 'transversetemporal'}
                frontal_lobe = {'caudalmiddlefrontal', 'lateralorbitofrontal', 'paracentral', 'parsopercularis',
                                'parsorbitalis', 'parstriangularis', 'precentral', 'rostralmiddlefrontal',
                                'superiorfrontal', 'medialorbitofrontal', 'frontalpole'}
                parietal_lobe = {'inferiorparietal', 'postcentral', 'precuneus', 'superiorparietal', 'supramarginal'}
                occipital_lobe = {'cuneus', 'lateraloccipital', 'pericalcarine', 'lingual'}
                cingulate_cortex = {'caudalanteriorcingulate', 'isthmuscingulate', 'posteriorcingulate', 'rostralanteriorcingulate'}
                insula_cortex = {'insula'}
                check = col.split('_')[2].split('-')[0]
                if check in temporal_lobe:
                    return 'Temporal lobe'
                elif check in frontal_lobe:
                    return 'Frontal lobe'
                elif check in parietal_lobe:
                    return 'Parietal lobe'
                elif check in occipital_lobe:
                    return 'Occipital lobe'
                elif check in cingulate_cortex:
                    return 'Cingulate cortex'
                elif check in insula_cortex:
                    return 'Insula cortex'
                else:
                    return 'Other'
            elif 'subcor' == col.split('_')[1]:
                return 'Subcortical region' # To do
            else:
                return 'DTI region' # To do
            
        # Columns: Feature derivatives
        _, X_col_names, _ = self.get_holdout_data(H5, group=False)
        COL = pd.DataFrame(
            {'Feature name': X_col_names}
        )
        COL['Modality'] = [i.split('_')[0] for i in COL['Feature name']]
        COL['Type'] = [type_check(i) for i in COL['Feature name']]
        COL['Lobe Region'] = [lobe_region(i) for i in COL['Feature name']]
        COL['Value'] = [i.split('-')[-1].split('_')[-1] for i in COL['Feature name']]
        
        # Columns: Mean and std derivatives
        for i in SHAP:
            Data = i.replace('.sav','')
            mean, _ = self.load_abs_SHAP(i)
            DF = pd.DataFrame(
                {f'{Data} mean': mean} 
            )
            COL = pd.concat([COL, DF], axis=1)
        
        for i in SHAP:
            Data = i.replace('.sav','')
            _, std = self.load_abs_SHAP(i)
            DF = pd.DataFrame(
                {f'{Data} std' : std}
            )
            COL = pd.concat([COL, DF], axis=1)
        
        # Columns: Mean and std of mean|SHAP|
        COL[f'GB_{SESSION} all mean'] = COL[COL.columns[5:12]].mean(axis=1)
        COL[f'LR_{SESSION} all mean'] = COL[COL.columns[12:19]].mean(axis=1)
        COL[f'SVM-lin_{SESSION} all mean'] = COL[COL.columns[19:26]].mean(axis=1)
        COL[f'SVM-rbf_{SESSION} all mean'] = COL[COL.columns[26:33]].mean(axis=1)
        COL[f'GB_{SESSION} all std'] = COL[COL.columns[5:12]].std(axis=1)
        COL[f'LR_{SESSION} all std'] = COL[COL.columns[12:19]].std(axis=1)
        COL[f'SVM-lin_{SESSION} all std'] = COL[COL.columns[19:26]].std(axis=1)
        COL[f'SVM-rbf_{SESSION} all std'] = COL[COL.columns[26:33]].std(axis=1)
        
        if save == True:
            save_path = f"{self.DATA_DIR}/posthoc/explainers/all_{SESSION}_SHAP.csv"
            if not os.path.isdir(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            COL.to_csv(save_path, index=None)
        return COL

    def to_sorted_mean_SHAP(self, DF, MODEL, SESSION, save=False):
        DF = DF.copy()
        rbf0 = [list(x) for x in zip(DF['Feature name'], DF[f'{MODEL}0_{SESSION} mean'], DF[f'{MODEL}0_{SESSION} std'])]
        rbf0.sort(key=lambda x:-x[1])
        rbf1 = [list(x) for x in zip(DF['Feature name'], DF[f'{MODEL}1_{SESSION} mean'], DF[f'{MODEL}1_{SESSION} std'])]
        rbf1.sort(key=lambda x:-x[1])
        rbf2 = [list(x) for x in zip(DF['Feature name'], DF[f'{MODEL}2_{SESSION} mean'], DF[f'{MODEL}2_{SESSION} std'])]
        rbf2.sort(key=lambda x:-x[1])
        rbf3 = [list(x) for x in zip(DF['Feature name'], DF[f'{MODEL}3_{SESSION} mean'], DF[f'{MODEL}3_{SESSION} std'])]
        rbf3.sort(key=lambda x:-x[1])
        rbf4 = [list(x) for x in zip(DF['Feature name'], DF[f'{MODEL}4_{SESSION} mean'], DF[f'{MODEL}4_{SESSION} std'])]
        rbf4.sort(key=lambda x:-x[1])
        rbf5 = [list(x) for x in zip(DF['Feature name'], DF[f'{MODEL}5_{SESSION} mean'], DF[f'{MODEL}5_{SESSION} std'])]
        rbf5.sort(key=lambda x:-x[1])
        rbf6 = [list(x) for x in zip(DF['Feature name'], DF[f'{MODEL}6_{SESSION} mean'], DF[f'{MODEL}6_{SESSION} std'])]
        rbf6.sort(key=lambda x:-x[1])
        
        DF[f'sorted {MODEL}0_{SESSION} name'] = [i[0] for i in rbf0]
        DF[f'sorted {MODEL}1_{SESSION} name'] = [i[0] for i in rbf1]
        DF[f'sorted {MODEL}2_{SESSION} name'] = [i[0] for i in rbf2]
        DF[f'sorted {MODEL}3_{SESSION} name'] = [i[0] for i in rbf3]
        DF[f'sorted {MODEL}4_{SESSION} name'] = [i[0] for i in rbf4]
        DF[f'sorted {MODEL}5_{SESSION} name'] = [i[0] for i in rbf5]
        DF[f'sorted {MODEL}6_{SESSION} name'] = [i[0] for i in rbf6]
        
        DF[f'sorted {MODEL}0_{SESSION} mean'] = [i[1] for i in rbf0]
        DF[f'sorted {MODEL}1_{SESSION} mean'] = [i[1] for i in rbf1]
        DF[f'sorted {MODEL}2_{SESSION} mean'] = [i[1] for i in rbf2]
        DF[f'sorted {MODEL}3_{SESSION} mean'] = [i[1] for i in rbf3]
        DF[f'sorted {MODEL}4_{SESSION} mean'] = [i[1] for i in rbf4]
        DF[f'sorted {MODEL}5_{SESSION} mean'] = [i[1] for i in rbf5]
        DF[f'sorted {MODEL}6_{SESSION} mean'] = [i[1] for i in rbf6]
        
        DF[f'sorted {MODEL}0_{SESSION} std'] = [i[2] for i in rbf0]
        DF[f'sorted {MODEL}1_{SESSION} std'] = [i[2] for i in rbf1]
        DF[f'sorted {MODEL}2_{SESSION} std'] = [i[2] for i in rbf2]
        DF[f'sorted {MODEL}3_{SESSION} std'] = [i[2] for i in rbf3]
        DF[f'sorted {MODEL}4_{SESSION} std'] = [i[2] for i in rbf4]
        DF[f'sorted {MODEL}5_{SESSION} std'] = [i[2] for i in rbf5]
        DF[f'sorted {MODEL}6_{SESSION} std'] = [i[2] for i in rbf6]
        if save == True:
            save_path = f"{self.DATA_DIR}/posthoc/explainers/sorted_{MODEL}_{SESSION}_SHAP.csv"
            if not os.path.isdir(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            DF.to_csv(save_path, index=None)
        return DF
    
    def to_SHAP(self, SHAP, NAME, save=False):
        """ Load the SHAP file
        
        Parameters
        ----------
        SHAP : list
            List of the SHAP files, all_model*_session*_SHAP (*.csv)
        
        Returns
        -------
        DF : pandas.dataframe
            The unified SHAP file (*.csv)
        
        """
        DF = pd.concat(SHAP)
        if save == True:
            save_path = f"{self.DATA_DIR}/posthoc/explainers/{NAME}"
            if not os.path.isdir(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            DF.to_csv(save_path, index=None)
        return DF
    
    def read_INSTRUMENT(self, instrument_file):
        """ Load the Instruemnt file
        
        Parameters
        ----------
        instrument_file : string
            instrument file
            
        Returns
        -------
        self.INSTRUMENT : pandas.dataframe
            The IMAGEN's instrument file (*.csv)
            
        Notes
        -----
        This function cover the 8 ROIs:
        Psychological profile - NEO, SURPS,
        Socio-economic profile - CTQ, CTS, LEQ, PBQ, GEN,
        Other co-morbidities - FTND
            
        Example
        -------
        >>> from imagen_posthocloader import *
        >>> DATA = IMAGEN_posthoc()
        >>> DF = DATA.read_INSTURMENT(
        ...     instrument_file)            # instrument file
        >>> DF_FU3 = DF.groupby('Session').get_groupby('FU3')
        
        """     
        instrument_path = f"{self.DATA_DIR}/posthoc/{instrument_file}"
        DF = pd.read_csv(instrument_path, low_memory=False)
        self.INSTRUMENT = DF
        return self.INSTRUMENT

    def read_HDF5(self, hdf5_file):
        """ Load the HDF5 file
        
        Parameters
        ----------
        hdf5_file : string,
            The IMAGEN's h5df file (*.csv)
        
        Returns
        -------
        self.HDF5 : pandas.dataframe
            The slected HDF5 dataframe
            
        Examples
        --------
        >>> from imagen_posthocloader import *
        >>> DATA = IMAGEN_posthoc()
        >>> HDF5 = DATA.read_HDF5(
        ...     h5py_file)                               # HDF5
        >>> HDF5_FU3 = HDF5.groupby('Session').get_group('fu3')

        Notes
        -----
        Dataset:
        Training and Holdout
        """
        # Load the hdf5 file
        hdf5_path = f"{self.DATA_DIR}/posthoc/{hdf5_file}"
        DF = pd.read_csv(hdf5_path, low_memory=False)
        self.HDF5 = DF
        return self.HDF5
  
    def read_RUN(self, run_file):
        """ Load the RUN file
        
        Parameters
        ----------
        run_file : string
            ML models result run.csv path
            
        Returns
        -------
        self.RUN : pandas.dataframe
            The RUN dataframe
        
        Examples
        --------
        >>> from imagen_posthocloader import *
        >>> DATA = IMAGEN_posthoc()
        >>> RUN = DATA.read_RUN(
        ...     run_file)                               # run
        >>> RUN_FU3 = RUN.groupby('Session').get_group('fu3')
        
        Notes
        -----
        There are more options to select ROI columns
        General information:
            'ID','Session','Trial','dataset','io','technique','Model',
            'TP prob','TN prob','FP prob','FN prob','T prob','F prob','Prob',
            'Predict TF','Model PN','Label PN','true_label','prediction'
        
        """
        # Load the hdf5 file
        run_path = f"{self.DATA_DIR}/posthoc/{run_file}"
        DF = pd.read_csv(run_path, low_memory=False)
        self.RUN = DF
        return self.RUN
    
    def read_SHAP(self, SHAP_file):
        """ Load the SHAP file
        
        Parameters
        ----------
        SHAP_file : string
            SHAP file
            
        Returns
        -------
        self.SHAP : pandas.dataframe
            The SHAP file (*.csv)
        
        Example
        -------
        >>> from imagen_posthocloader import *
        >>> DATA = SHAP_loader()
        >>> DF = DATA.read_SHAP(
        ...      SHAP_file)               # SHAP file
        
        """
        SHAP_path = self.DATA_DIR+"/posthoc/explainers/"+SHAP_file
        DF = pd.read_csv(SHAP_path, low_memory=False)
        self.SHAP = DF
        return self.SHAP

    def to_posthoc(self, DATA, save=False):
        """ Set the Posthoc file
        
        Parameters
        ----------        
        DATA : list
            [INSTRUMENT.csv,                   # instrument
             HDF5.csv,                         # hdf5
             RUN.csv]                          # run
             
        save : boolean
            if save == True, then save it as .csv

        Returns
        -------
        self.posthoc : pandas.dataframe
            The Psothoc dataframe

        Examples
        --------
        >>> from imagen_psothocloader import *
        >>> Posthoc = IMAGEN_posthoc()
        >>> DF = Posthoc.to_posthoc(
        ...     DATA)               # INSTRUMENT, HDF5, RUN                   
        >>> DF_FU3 = DF.groupby('Session').get_group('FU3')

        Notes
        -----
        ONLY FU3 implementation, other seession need modification
        
        """
        HDF5 = self.read_HDF5(DATA[0])
        INST = self.read_INSTRUMENT(DATA[1])
        RUN = self.read_RUN(DATA[2])
        
        H_FU3 = HDF5.groupby('Session').get_group('FU3')
        I_FU3 = INST.groupby('Session').get_group('FU3')
        R_FU3 = RUN.groupby('Session').get_group('FU3')
        HR_FU3 = pd.merge(H_FU3, R_FU3, on=['ID','Session'], how='left')
        DF = pd.merge(HR_FU3, I_FU3, on=['ID','Session'], how='left')
        self.posthoc = DF
        
        if save == True:
            save_path = f"{self.DATA_DIR}/posthoc/IMAGEN_posthoc.csv"
            if not os.path.isdir(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            DF.to_csv(save_path, index=None)
        return self.posthoc

    def read_posthoc(self, posthoc_file):
        """ Load the Posthoc file
        
        Parameters
        ----------
        diagnosis_file : string
            diagnosis file path
            
        Returns
        -------
        self.posthoc : pandas.dataframe
            The Posthoc dataframe
        
        Examples
        --------
        >>> from imagen_psothocloader import *
        >>> Posthoc = IMAGEN_posthoc()
        >>> DF = Posthoc.read_posthoc(
        ...     DATA)               # INSTRUMENT, HDF5, RUN                   
        >>> DF_FU3 = DF.groupby('Session').get_group('FU3')

        Notes
        -----
        ONLY FU3 implementation, other seession need modification
        
        """
         # Load the posthoc file
        run_path = f"{self.DATA_DIR}/posthoc/{posthoc_file}"
        DF = pd.read_csv(run_path, low_memory=False)
        self.posthoc = DF
        return self.posthoc

#     def __str__(self):
#         """ Print the instrument loader steps """
#         return "Step 1. load the phenotype: " \
#                + str(self.h5py_file.replace(".h5", "")) \
#                + "\n        Class = " + str(list(self.d.keys())[0]) \
#                + ", n = " + str(len(self.ALL)) + " (HC = " \
#                + str(len(self.HC)) + ", AAM = " + str(len(self.AAM)) +")" \
#                + "\n" + "Step 2. load the instrument dataset: " \
#                + str(self.instrument_file.replace(".csv",'')) \
#                + "\n" + "Step 3. generate the " + str(self.SESSION) +"_" \
#                + str(self.h5py_file.replace(".h5", "")) \
#                + str(self.instrument_file.replace("IMAGEN-IMGN", "")) \
#                + "\n        The dataset contains " + str(self.NEW_DF.shape[0]) \
#                + " samples and " + str(self.NEW_DF.shape[1]) + " columns" \
#                + "\n" + "Step 4. select " + str(self.SESSION) +"_" \
#                + str(self.h5py_file.replace(".h5", "")) \
#                + str(self.instrument_file.replace("IMAGEN-IMGN", "")) \
#                + "\n        The dataset contains " + str(self.NEW_DF.shape[0]) \
#                + " samples and " + str(self.NEW_DF.shape[1]) + " columns" \
#                + "\n        Variables = " + str(self.Variables)
#################################################################################