#################################################################################
""" IMAGEN Instrument Summary Statistic """
# Author: JiHoon Kim, <jihoon.kim@fu-berlin.de>, 16th August 2021
#
import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro, levene, ttest_ind, bartlett
from statannot import add_stat_annotation

class IMAGEN_descriptive:
    """ Plot the demographic statistics """
    def __init__(self, DF, COL):
        """ Set up the Dataframe and Columns
        
        Parameters
        ----------
        DF : pandas.dataframe
            The Instrument dataframe
            
        COL : string
            Instruments columns: ROI Columns, Sex, Site, Class
        
        """
        self.DF = DF
        self.Columns = list(COL[:-6])
        self.Target = list(COL[-1:])

    def histogram(self, bins=False, save=False):
        """ Plot the histogram
        
        Parameters
        ----------
        bins : Boolean, optional
            True : default (10), False : Sturge's Rule
        
        save : Boolean, optional
            If it is True then save as file (*.png)
        
        Notes
        -------
        Plot the subplot of the Histogram
        
        Examples
        --------
        >>> from imagen_plot_statistics import *
        >>> plot_binge_FU3_NEO = plot_demographic(df_binge_FU3_NEO,
        >>>                                       col_binge_FU3_NEO)
        >>> plot_binge_FU3_NEO.histogram(bins=True, save=False)

        """
        # Set the columns
        columns = self.Columns
        # Compute the bins based on Sturge's Rule
        b = 1 + (3.3*math.log(len(self.DF)))
        k = 10 if bins == False else math.ceil(b)
        # Plot the histogram
        self.DF[columns].hist(bins = k, figsize=(20, 14))
        
        if save == True:
            # PDF print version function needed
            pass
    
    def pairplot(self, save=False):
        """ Plot the pairplot

        Parameters
        ----------
        save : Boolean, optional
            If it is True then save as file (*.png)

        Notes
        -----
        Plot the subplot of the pairplot

        Examples
        --------
        >>> from imagen_plot_statistics import *
        >>> plot_binge_FU3_NEO = plot_demographic(df_binge_FU3_NEO,
        >>>                                       col_binge_FU3_NEO)
        >>> plot_binge_FU3_NEO.pairplot(save=False)
        
        """
        columns = self.Columns
        # Plot the pairplot
        sns.pairplot(data=self.DF, vars=columns, hue='Class',
                     plot_kws={'alpha': 0.2}, height=3,
                     diag_kind='kde', palette="Set1")
        
        if save == True:
            # PDF print version function needed
            pass
    
    def violinplot(self, save=False):
        """ Plot the violinplot

        Parameters
        ----------
        save : Boolean, optional
            If it is True then save as file (*.png)
        
        Notes
        -----
        Plot the subplot of the violinplot
        
        Examples
        --------
        >>> from imagen_plot_statistics import *
        >>> plot_binge_FU3_NEO = plot_demographic(df_binge_FU3_NEO,
        >>>                                       col_binge_FU3_NEO)
        >>> plot_binge_FU3_NEO.violinplot()
        
        """
        columns  = self.Columns       
        title = self.Columns
        
        # violin plot
        sns.set(style="whitegrid", font_scale=1.5)
        fig, axes = plt.subplots(nrows=1, ncols=len(columns)+1,
                                 figsize=((len(columns)+1)**2, len(columns)+1))

        sns.countplot(x="Class", hue='Sex', order=['HC', 'AAM'],
                      data = self.DF, ax = axes[0], palette="Set2")

        for i, j in enumerate(columns):
            axes[i+1].set_title(title[i])
            sns.violinplot(x="Class", y=j, data=self.DF, order=['HC', 'AAM'],
                           inner="quartile", ax = axes[i+1], palette="Set1")
            add_stat_annotation(ax = axes[i+1], data=self.DF, x="Class", y=j,
                                box_pairs = [("HC","AAM")], order=["HC","AAM"], test='t-test_ind',
                                text_format='star', loc='inside')
                
        # violin plot
        sns.set(style="whitegrid", font_scale=1.5)
        fig, axes = plt.subplots(nrows=1, ncols=len(columns)+1,
                                     figsize=((len(columns)+1)**2, len(columns)+1))

        sns.countplot(x="Class", hue='Sex', data = self.DF, order=['HC', 'AAM'],
                          ax = axes[0], palette="Set2")

        for i, j in enumerate(columns):
            axes[i+1].set_title(title[i])
            sns.violinplot(x="Class", y=j, hue='Sex', data=self.DF,
                           order=['HC', 'AAM'], inner="quartile",
                           ax = axes[i+1], split=True, palette="Set2")

    def catplot(self, save=False):
        """ Plot the catplot

        Parameters
        ----------
        save : Boolean, optional
            If it is True then save as file (*.png)
        
        Notes
        -----
        Plot the subplot of the catplot

        Examples
        --------
        >>> from imagen_plot_statistics import *
        >>> plot_binge_FU3_NEO = plot_demographic(df_binge_FU3_NEO,
        >>>                                       col_binge_FU3_NEO)
        >>> plot_binge_FU3_NEO.catplot(save=False)
        
        """
        columns = self.Columns
        
        # catplot
        sns.set(style="whitegrid", font_scale=1.5)

        for i, j in enumerate(columns):
            sns.catplot(x='Sex', y=j, hue = 'Class', col='Site', 
                        inner="quartile", data = self.DF, kind='violin',
                        split=True, height=4, aspect=.7, palette="Set2")
            
    def categorical_plot(self, save=False):
        """ Plot the barplot # change it lineplot
        
        Parameters
        ----------
        save : Boolean, optional
            If it is True then save as file (*.png)
        
        Notes
        -----
        Plot the subplot of the barplot        
        
        Examples
        --------
        >>> from imagen_plot_statistics import *
        >>> plot_binge_FU3_NEO = plot_demographic(df_binge_FU3_NEO,
        >>>                                       col_binge_FU3_NEO)
        >>> plot_binge_FU3_NEO.barplot(save=False)
        
        """
        columns = self.Columns
        # Table of nicotin dependence
        nd_class = pd.crosstab(index=self.DF["Class"], 
                               columns=self.DF["Nicotine dependence"],
                               margins=True)   # Include row and column totals

        nd_class.columns = ["highly dependent", "less dependent","moderately dependent", "coltotal"]
        nd_class.index= ["AAM","HC","rowtotal"]
        freq_nd_class = nd_class/nd_class.loc["rowtotal","coltotal"]

        print(f"{nd_class} \n \n {freq_nd_class} \n")
        
        ax = sns.countplot(y=columns[0], hue="Class", data=self.DF, palette="Set2")
        
        s = sns.catplot(y=columns[0], hue="Class", col="Sex", palette="Set2",
                        data=self.DF, kind="count", height=4, aspect=.7);
        
        c = sns.catplot(y=columns[0], hue="Class", col="Site", palette="Set2",
                        data=self.DF, kind="count", height=4, aspect=.7);
        
        
    def to_pdf(self):
        pass
    
    def demographic_plot(self, bins=False, save=False, viz=False):
        """ Plot the Summary Statistics
        
        Parameters
        ----------
        bins : Boolean, optional
            True : default (10), False : Sturge's Rule
        
        save : Boolean, optional
            If it is True then save as file (*.png)
        
        Notes
        -------
        Plot the Summary Statistics.
        Later divided into one Categorical way, the other numerical

        Examples
        --------
        >>> from imagen_plot_statistics import *
        >>> plot_binge_FU3_NEO = plot_demographic(df_binge_FU3_NEO, col_binge_FU3_NEO)
        >>> plot_binge_FU3_NEO.demographic_plot()

        """
        # Plot the demographic
        self.histogram(bins, save)
        self.pairplot(save)
        self.violinplot(save)
        self.catplot(save)
#         self.barplot(save)
        
        if viz == True:
            print(f"{'-'*83} \n{self.__str__()} \n{'-'*83}")
    
    def __str__(self):
        """ Print the Summary statistics """
        return 'Plot 1. histogram: '+ str(self.Columns) \
               +"\n"+'Plot 2. pariplot: '+ str(self.Columns) \
               +'\n'+'Plot 3. violinplot: '+ str(self.Columns) \
               +'\n'+'Plot 4. catplot: '+ str(self.Columns) \
               +'\n'+'Plot 5. barplot: '+ str(self.Columns)
    
class IMAGEN_inference(IMAGEN_descriptive):
    """ Compute the inference statistics """
    def __init__(self, DF, COL):
        """ Set up the Dataframe and Columns
        
        Parameters
        ----------
        DF : pandas.dataframe
            The Instrument dataframe
            
        COL : string
            Instruments columns: ROI Columns, Sex, Site, Class
        
        """
        self.DF = DF
        self.Columns = list(COL[:-6])
        self.Target = list(COL[-1:])
        
    def inference_statistics(self):
        for mean in self.Columns:
            print("-"*10, mean)
            myAAM = list(self.DF[self.DF['Class'] == 'AAM'][mean].values)
            AAM = [x for x in myAAM if pd.isnull(x) == False]
            myHC = list(self.DF[self.DF['Class'] == 'HC'][mean].values)    
            HC = [x for x in myHC if pd.isnull(x) == False]
            
            # Shapiro-Wilks
            normal1 = shapiro(AAM)
            normal2 = shapiro(HC)
            # Levene test
            normal3 = levene(AAM,HC)
            # bartlett test
            variance = bartlett(AAM, HC)
            # ttest
            ttest1 = ttest_ind(AAM, HC)
            ttest2 = ttest_ind(AAM, HC, equal_var=False)
            print(f'Shapiro-Wilks AAM: {normal1} \n'
                  f'Shapiro-Wilks HHC: {normal2} \n'
                  f'Levene test:       {normal3} \n'
                  f'Bartlett test:     {variance} \n'
                  f'T test:            {ttest1} \n'
                  f'T test:            {ttest2} \n')
            
    def ANOVA(self):
        pass

    def chi_squared(self):
        pass
    
    def __str__(self):
        """ Print the Inference statistics"""
        return 'Compute 1. normality check: '+ str(self.Columns) \
               +"\n"+'compute 2. t-test: '+ str(self.Columns)# \
               #+'\n'+'Compute 3. ANOVA: '+ str(self.Columns) \
               #+'\n'+'Compute 4. chi_squared: '+ str(self.Columns)
    
class IMAGEN_statistics(IMAGEN_inference):
    """ Summary of Descriptive, Inference Statistics """
    def __init__(self, DF, COL):
        """ Set up the Dataframe and Columns
        
        Parameters
        ----------
        DF : pandas.dataframe
            The Instrument dataframe
            
        COL : string
            Instruments columns: ROI Columns, Sex, Site, Class
        
        """
        self.DF = DF
        self.Columns = list(COL[:-6])
        self.Target = list(COL[-1:])
        
    def to_statistics(self):
        pass