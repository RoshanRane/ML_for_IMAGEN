#################################################################################
""" IMAGEN Instrument Loader using H5DF file in all Session """
# Author: JiHoon Kim, <jihoon.kim@fu-berlin.de>, 7th September 2021
#
import os
import h5py
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class Instrument_loader:
    def __init__(self, DATA_DIR="/ritter/share/data/IMAGEN"):
        """ Set up path
        
        Parameters
        ----------
        DATA_DIR : string, optional
            Directory IMAGEN absolute path
        
        """
        # Set the directory path: IMAGEN
        self.DATA_DIR = DATA_DIR
        
    def load_HDF5(self, h5py_file):
        """ Generate the list,
        all subject (ALL), healthy control (HC),
        adolscent alcohol misuse (AAM), Sex, Site, and Class
        
        Parameters
        ----------
        h5py_file : string,
            The IMAGEN's h5df file name (*.h5)
            
        Notes
        -----
        It contain many list for self.__str__().
            self.h5py_file, self.d, self.ALL, self.HC, self.AAM,
            self.SEX, self.SITE, self.CLASS

        """     
        # If OSError, export HDF5_USE_FILE_LOCKING='FALSE'
        # set the h5 file absolute path
        self.h5py_file = h5py_file
        h5py_absolute_path = f'{self.DATA_DIR}/h5files/{self.h5py_file}'
        
        # Convert h5py to list
        self.d = h5py.File(h5py_absolute_path, 'r')
        # Set ALL, HC, and AAM
        b_list = list(np.array(self.d[list(self.d.keys())[0]]))
        self.ALL = list(np.array(self.d['i']))
        self.HC = [self.ALL[i] for i, j in enumerate(b_list) if j%2==0]
        self.AAM = [self.ALL[i] for i, j in enumerate(b_list) if j%2==1]
        # Set Sex
        sex = list(np.array(self.d['sex']))
        self.SEX = ['Male' if i==0 else 'Female' for i in sex]
        # Set Site
        sites = list(np.array(self.d['site']))
        center = {0: 'Paris', 1: 'Nottingham', 2:'Mannheim', 3:'London',
                  4: 'Hamburg', 5: 'Dublin', 6:'Dresden', 7:'Berlin'}
        self.SITE = [center[i] for i in sites]
        # Set Class
        target = list(np.array(self.d[list(self.d.keys())[0]]))
        self.CLASS = ['HC' if i==0 else 'AAM' for i in target]

    def load_INSTRUMENT(self, SESSION, DATA, instrument_file):
        """ Load the INSTRUMENT file
        
        Parameters
        ----------        
        SESSION : string
            One of the four SESSION (BL, FU1, FU2, FU3)
            
        DATA : string
            The research of interest Instrument Name
            
        instrument_file : string
            The IMAGEN's instrument file (*.csv)

        Returns
        -------
        self.DF : pandas.dataframe
            The Instrument dataframe
            
        Notes
        -----
        Not only load the instrument file but also check the ID.
        And return dataframe index as ID.
            
        """
        # Set Session, ROI data, and Instrument file
        self.SESSION = SESSION
        self.DATA = DATA
        self.instrument_file = instrument_file
        # Load the instrument file       
        instrument_path = f"{self.DATA_DIR}/IMAGEN_RAW/2.7/{self.SESSION}"+\
                            f"/psytools/{self.instrument_file}"
        DF = pd.read_csv(instrument_path, low_memory=False)
        # Add the column : ID
        def usercode(x):
            return int(x[:12])
        DF['ID'] = DF['User code'] if self.SESSION=='FU3' \
        else DF['User code'].apply(usercode)
        self.DF = DF
        return self.DF

    def generate_new_instrument(self, save=False, viz=False):
        """ Generate the new instrument,
        Rows: Subjects by ID HDF5 & INSTRUMENT
        Cols: ROI Columns, Data, Session, ID, Sex, Site, Class (HC, AAM)

        Parameters
        ----------   
        save : Boolean, optional
            If it is true, save to *.csv
        
        viz : Boolean, optional
            If it is true, print the steps
        
        Returns
        -------        
        self.NEW_DF : pandas.dataframe
            The new instrument file (*.csv)
        
        self.Variables : string
            Instruments columns: ROI Columns, Data, Session, ID, Sex, Site, Class
        
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
        # -------------------------------------- #
        # ID, Sex, Site, Class columns           #
        # -------------------------------------- #
        DF_2 = self.DF.set_index('ID').reindex(self.ALL)
        DF_2['Data'] = self.h5py_file[:-3]
        DF_2['Session'] = self.SESSION
        DF_2['ID'] = DF_2.index
        DF_2['Sex'] = self.SEX
        DF_2['Site'] = self.SITE
        DF_2['Class'] = self.CLASS
        
        # -------------------------------------- #
        # ROI Columns: Psychological profile     #
        # -------------------------------------- #
        if self.DATA == 'NEO':
            # Rename the columns
            NEW_DF = DF_2.rename(
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
            self.NEW_DF = NEW_DF[self.Variables]
        
        if self.DATA == 'SURPS':
            # Rename the columns
            NEW_DF = DF_2.rename(
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
            self.NEW_DF = NEW_DF[self.Variables]

        # -------------------------------------- #
        # ROI Columns: Socio-economic profile    #
        # -------------------------------------- #
        if self.DATA == 'CTQ':
            emot_abu = ['CTQ_3','CTQ_8','CTQ_14','CTQ_18','CTQ_25']
            phys_abu = ['CTQ_9','CTQ_11','CTQ_12','CTQ_15','CTQ_17']
            sexual_abu = ['CTQ_20','CTQ_21','CTQ_23','CTQ_24','CTQ_27']
            emot_neg = ['CTQ_5','CTQ_7','CTQ_13','CTQ_19','CTQ_28']
            phys_neg = ['CTQ_1','CTQ_2','CTQ_4','CTQ_6','CTQ_26']
            denial = ['CTQ_10','CTQ_16','CTQ_22']
            # Generate the columns
            DF_2['Emotional abuse sum'] = DF_2[emot_abu].sum(axis=1,skipna=False)
            DF_2['Physical abuse sum'] = DF_2[phys_abu].sum(axis=1,skipna=False)
            DF_2['Sexsual abuse sum'] = DF_2[sexual_abu].sum(axis=1,skipna=False)
            DF_2['Emotional neglect sum'] = DF_2[emot_neg].sum(axis=1,skipna=False)
            DF_2['Physical neglect sum'] = DF_2[phys_neg].sum(axis=1,skipna=False)
            DF_2['Denial sum'] = DF_2[denial].sum(axis=1, skipna=False)
            NEW_DF = DF_2
            # Set the roi variables
            self.Variables = [
                'Emotional abuse sum','Physical abuse sum','Sexsual abuse sum',
                'Emotional neglect sum','Physical neglect sum','Denial sum',
                'Data','Session','ID','Sex','Site','Class'
            ]
            self.NEW_DF = NEW_DF[self.Variables]

        if self.DATA == 'LEQ':
            # Rename the columns
            NEW_DF = DF_2.rename(
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
            self.NEW_DF = NEW_DF[self.Variables]

        if self.DATA == 'PBQ':
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
            DF_2["pbq_03"] = DF_2['pbq_03'].apply(test)
            DF_2['pbq_03a'] = DF_2['pbq_03a'].apply(day)
            DF_2['pbq_03b'] = DF_2['pbq_03b'].apply(age)
            DF_2["pbq_03c"] = DF_2['pbq_03c'].apply(cigarettes)
            DF_2["pbq_05"] = DF_2['pbq_05'].apply(test)
            DF_2["pbq_05a"] = DF_2['pbq_05a'].apply(cigarettes)
            DF_2["pbq_05b"] = DF_2['pbq_05b'].apply(cigarettes)
            DF_2["pbq_05c"] = DF_2['pbq_05c'].apply(cigarettes)
            DF_2["pbq_06"] = DF_2['pbq_06'].apply(test)
            DF_2["pbq_06a"] = DF_2['pbq_06a'].apply(cigarettes)
            DF_2["pbq_12"] = DF_2['pbq_12'].apply(test)
            DF_2["pbq_13"] = DF_2['pbq_13'].apply(test)
            DF_2["pbq_13a"] = DF_2['pbq_13a'].apply(alcohol)
            DF_2["pbq_13b"] = DF_2['pbq_13b'].apply(drinks)
            DF_2["pbq_13g"] = DF_2['pbq_13g'].apply(stage)    
            # Rename the columns
# It can be            
            NEW_DF = DF_2
            # Set the roi variables
            self.Variables = [
                'pbq_03','pbq_03a','pbq_03b','pbq_03c','pbq_05',
                'pbq_05a','pbq_05b','pbq_05c','pbq_06','pbq_06a',
                'pbq_12','pbq_13','pbq_13a','pbq_13b','pbq_13g',
                'Data','Session','ID','Sex', 'Site', 'Class'
            ]
            Check_DF = NEW_DF[self.Variables]
            # Exclude the row
            # Duplicate ID: 71766352, 58060181, 15765805, 15765805 in FU1
            # Different ID: 12809392 in both BL and FU1
            for i in [71766352, 58060181, 15765805, 12809392]:
                is_out = (Check_DF['ID']==i) & (Check_DF['Session']=='FU1')
                Check_DF = Check_DF[~is_out]
            for i in [12809392]:
                is_out = (Check_DF['ID']==i) & (Check_DF['Session']=='BL')
                Check_DF = Check_DF[~is_out]
            self.NEW_DF = Check_DF

        if self.DATA == 'CTS':
            # Rename the columns
            NEW_DF = DF_2.rename(
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
            self.NEW_DF = NEW_DF[self.Variables]
        
        if self.DATA == 'GEN':
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
            DF_2['Disorder_PF_1'] = DF_2['Disorder_PF_1'].apply(disorder)
            DF_2['Disorder_PF_2'] = DF_2['Disorder_PF_2'].apply(disorder)
            DF_2['Disorder_PF_3'] = DF_2['Disorder_PF_3'].apply(disorder)
            DF_2['Disorder_PF_4'] = DF_2['Disorder_PF_4'].apply(disorder)
            DF_2['Disorder_PM_1'] = DF_2['Disorder_PM_1'].apply(disorder)
            DF_2['Disorder_PM_2'] = DF_2['Disorder_PM_2'].apply(disorder)
            DF_2['Disorder_PM_3'] = DF_2['Disorder_PM_3'].apply(disorder)
            DF_2['Disorder_PM_4'] = DF_2['Disorder_PM_4'].apply(disorder)
            DF_2['Disorder_PM_5'] = DF_2['Disorder_PM_5'].apply(disorder)
            DF_2['Disorder_PM_6'] = DF_2['Disorder_PM_6'].apply(disorder)
            NEW_DF = DF_2
            Variables = [
                'Disorder_PF_1','Disorder_PF_2','Disorder_PF_3','Disorder_PF_4',
                'Disorder_PM_1','Disorder_PM_2','Disorder_PM_3','Disorder_PM_4',
                'Disorder_PM_5','Disorder_PM_6','Session','ID','Data','Sex','Site','Class'
            ]
            Check_DF = NEW_DF[Variables]
            Check_DF['Paternal_disorder'] = Check_DF.loc[:, Check_DF.columns[:4]].apply(
                lambda x: ','.join(x.dropna().astype(str)), axis=1)
            Check_DF['Maternal_disorder'] = Check_DF.loc[:, Check_DF.columns[4:9]].apply(
                lambda x: ','.join(x.dropna().astype(str)), axis=1)

            # Set the roi variables
            self.Variables = [
                'Paternal_disorder','Maternal_disorder',
                'Session','ID','Data','Sex','Site','Class'
            ]
            self.NEW_DF = Check_DF[self.Variables]
            
        # -------------------------------------- #
        # ROI Columns: Other co-morbidities      #
        # -------------------------------------- #
        if self.DATA == 'FTND':
            # Generate the columns
            def test(x):
                if (7<=x and x <=10): return 'highly dependent'
                elif (4<=x and x <=6): return 'moderately dependent'
                elif (x<4): return 'less dependent'
                else: return np.NaN
            DF_2["Nicotine dependence"] = DF_2['ftnd_sum'].apply(test)
            NEW_DF = DF_2
            # Set the roi variables
            self.Variables = [
                'Nicotine dependence','Data','Session','ID','Sex','Site','Class'
            ]
            self.NEW_DF = NEW_DF[self.Variables]

#         def DAST_SESSION(SESSION):
#             if SESSION == 'FU3':
#                 Variables = ['sex', 'site', 'class']
#                 DATA_DF = self.NEW_DF[Variables]
#                 return Variables, DATA_DF
#             if 'DAST' == self.DATA: # 'DAST'
#                 self.VARIABLES, self.NEW_DF2 = DAST_SESSION(self.SESSION)


#         def SCID_SESSION(SESSION):
#             if SESSION == 'FU3':
#                 Variables = ['sex', 'site', 'class']
#                 DATA_DF = self.NEW_DF[Variables]
#                 return Variables, DATA_DF
#             if 'SCID' == self.DATA: # 'SCID'
#                 self.VARIABLES, self.NEW_DF2 = SCID_SESSION(self.SESSION)


#         def DMQ_SESSION(SESSION):
#             if SESSION == 'FU3':
#                 Variables = ['sex', 'site', 'class']
#                 DATA_DF = self.NEW_DF[Variables]
#                 return Variables, DATA_DF
#             if 'DMQ' == self.DATA: # 'DMQ'
#                 self.VARIABLES, self.NEW_DF2 = DMQ_SESSION(self.SESSION)


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
#                 DF_BSI = self.NEW_DF
#                 DF_BSI['somatization_mean'] = DF_CTQ[Somatization_labels].mean(axis=1,
#                                                                                skipna=False)
#                 DF_BSI['obsession_compulsion_mean'] = DF_CTQ[Somatization_labels].mean(axis=1,
#                                                                                skipna=False)             
#         elif 'BSI' == self.DATA: # 'BSI'
#             self.VARIABLES, self.NEW_DF2 = BSI_SESSION(self.SESSION)             
#                 Variables = Somatization_labels + Obsession_compulsion_labels \
#                         + Interpersonal_sensitivity_labels + Depression_labels \
#                         + Anxiety_labels + Hostility_labels \
#                         + Phobic_anxiety_labels + Paranoid_ideation_labels \
#                         + Psychoticism_labels + ['sex', 'site', 'class']
#                 DATA_DF = self.NEW_DF[Variables]
#                 return Variables, DATA_DF


#         def AUDIT_SESSION(SESSION):
#             if SESSION == 'FU3':
#                 Variables = ['sex', 'site', 'class']
#                 DATA_DF = self.NEW_DF[Variables]
#                 return Variables, DATA_DF
#         elif 'AUDIT' == self.DATA: # 'AUDIT'
#             self.VARIABLES, self.NEW_DF2 = AUDIT_SESSION(self.SESSION)


#         def MAST_SESSION(SESSION):
#             if SESSION == 'FU3':
#                 Variables = ['sex', 'site', 'class']
#                 DATA_DF = self.NEW_DF[Variables]
#                 return Variables, DATA_DF
#         elif 'MAST' == self.DATA: # 'MAST'
#             self.VARIABLES, self.NEW_DF2 = MAST_SESSION(self.SESSION)  
#         self.NEW_DF = NEW_DF[self.Variables]
#################################################################################
    
        if save == True:
            phenotype = self.h5py_file.replace(".h5", "")
            save_absolute_path = f"{self.DATA_DIR}/Instrument/"+\
                                 f"{phenotype}_{self.SESSION}_{self.DATA}.csv"
            # set the save option
            if not os.path.isdir(os.path.dirname(save_absolute_path)):
                os.makedirs(os.path.dirname(save_absolute_path))
            self.NEW_DF.to_csv(save_absolute_path)
            
        if viz == True:
            print(f"{'-'*83} \n{self.__str__()} \n{'-'*83}")
            print(f"{self.NEW_DF.info(), self.NEW_DF.describe()}")
        return self.NEW_DF

    def get_instrument(self, h5py_file, SESSION, instrument_file,
                       DATA, save=False, viz=False):
        """ Load the HDF5, INSTRUMENT. And generate the new instruemnt

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
        self.NEW_DF : pandas.dataframe
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
        self.load_HDF5(h5py_file)
        self.load_INSTRUMENT(SESSION, DATA, instrument_file)
        self.generate_new_instrument(save, viz)
        return self.NEW_DF
    
    def __str__(self):
        """ Print the instrument loader steps """
        return "Step 1. load the phenotype: " \
               + str(self.h5py_file.replace(".h5", "")) \
               + "\n        Class = " + str(list(self.d.keys())[0]) \
               + ", n = " + str(len(self.ALL)) + " (HC = " \
               + str(len(self.HC)) + ", AAM = " + str(len(self.AAM)) +")" \
               + "\n" + "Step 2. load the instrument dataset: " \
               + str(self.instrument_file.replace(".csv",'')) \
               + "\n" + "Step 3. generate the " + str(self.SESSION) +"_" \
               + str(self.h5py_file.replace(".h5", "")) \
               + str(self.instrument_file.replace("IMAGEN-IMGN", "")) \
               + "\n        The dataset contains " + str(self.NEW_DF.shape[0]) \
               + " samples and " + str(self.NEW_DF.shape[1]) + " columns" \
               + "\n" + "Step 4. select " + str(self.SESSION) +"_" \
               + str(self.h5py_file.replace(".h5", "")) \
               + str(self.instrument_file.replace("IMAGEN-IMGN", "")) \
               + "\n        The dataset contains " + str(self.NEW_DF.shape[0]) \
               + " samples and " + str(self.NEW_DF.shape[1]) + " columns" \
               + "\n        Variables = " + str(self.Variables)
    
class IMAGEN_instrument(Instrument_loader):
    def __init__(self, DATA_DIR="/ritter/share/data/IMAGEN"):
        """ Set up path
        
        Parameters
        ----------
        DATA_DIR : string, optional
            Directory IMAGEN absolute path
        
        """
        self.DATA_DIR = DATA_DIR
    
    def to_instrument(self, merge_list, merge_key, save=False):
        """ Merge the instrument files
        
        Parameters
        ----------
        merge_list : list
            The IMAGEN instruments list
            
        merge_key : list
            The IMAGEN instruments group name: BL, FU1, FU2, FU3
        
        save : Boolean, optional
            If it is true, save the file

        Returns
        -------        
        self.DF : pandas.dataframe
            The instrument file (*.csv)
        
        Examples
        --------
        >>> from imagen_instrumentloader import *
        >>> binge_NEO = IMAGEN_instrument()
        >>> df_binge_FU3_NEO = binge_NEO.to_instrument(
        ...     "IMAGEN-IMGN_NEO_FFI_FU3.csv",             # instrument
        ...     viz = True)                                # summary        
        
        """
        NEW2_DF = pd.concat(merge_list, keys=merge_key)
        if save == True:
            phenotype = self.h5py_file.replace(".h5", "")
            save_absolute_path = f"{self.DATA_DIR}/Instrument/"+\
                                 f"{phenotype}_{self.DATA}.csv"
            # set the save option
            if not os.path.isdir(os.path.dirname(save_absolute_path)):
                os.makedirs(os.path.dirname(save_absolute_path))
            NEW2_DF.to_csv(save_absolute_path, index=None)
        return NEW2_DF
        
    def read_instrument(self, instrument_path, viz=False):
        """ Select instrument file and get the columns
        
        Parameters
        ----------
        instrument_path : string
            The IMAGEN's instrument file (*.csv)
        
        viz : Boolean, optional
            If it is true, print the steps

        Returns
        -------        
        self.DF : pandas.dataframe
            The instrument file (*.csv)
        
        Examples
        --------
        >>> from imagen_instrumentloader import *
        >>> NEO = IMAGEN_instrument()
        >>> df_binge_FU3_NEO = NEO.read_instrument(
        ...     "IMAGEN-IMGN_NEO_FFI_FU3.csv", # instrument
        ...     viz = True)                    # summary        
        
        """
        self.instrument_path = instrument_path
        instrument_path = f"{self.DATA_DIR}/Instrument/{instrument_path}"
        self.DF = pd.read_csv(instrument_path, low_memory=False)
        self.VARIABLES = list(self.DF.columns)
        
        if viz == True:
            print(f"{'-'*83} \n{self.__str__()} \n{'-'*83}")
            print(f"{self.DF.info(), self.DF.describe()}")
        return self.DF
    
    def __str__(self):
        """ Print the instrument loader steps """
        return "Step 1. load the instrument: " \
               + "\n        File = " + str(self.instrument_path) \
               + "\n        The dataset contains " + str(self.DF.shape[0]) \
               + " samples and " + str(self.DF.shape[1]) + " columns" \
               + "\n        Variables = " + str(self.VARIABLES)