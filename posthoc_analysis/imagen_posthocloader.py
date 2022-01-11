#################################################################################
#!/usr/bin/env python
# coding: utf-8
""" IMAGEN Posthoc analysis Loader in all Session """
# Author: JiHoon Kim, <jihoon.kim@fu-berlin.de>, 11th January 2022
#
import os
import h5py
import shap
import pickle
import pandas as pd
import numpy as np
from glob import glob
from joblib import load
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

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
                d = h5py.File(path,'r')
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
        data = h5py.File(self.DATA_DIR+"/h5files/"+H5_DIR, 'r')
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
        data = h5py.File(self.DATA_DIR+"/h5files/"+H5_DIR, 'r')
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