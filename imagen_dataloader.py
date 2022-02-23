import pandas as pd
import numpy as np 
from os.path import join 
import os
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage.interpolation import zoom
import nibabel as nib
import h5py
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from slugify import slugify

from plotGraphs import plotGraph

DATA_DIR = "/ritter/share/data/IMAGEN/IMAGEN_RAW/2.7/"
BIDS_DIR = "/ritter/share/data/IMAGEN/IMAGEN_BIDS/"
# path to questionaires files
qs = dict(
        AUDIT_BL  = DATA_DIR + "BL/psytools/IMAGEN-IMGN_AUDIT_CHILD_RC5-IMAGEN_DIGEST.csv",
        AUDIT_FU1 = DATA_DIR + "FU1/psytools/IMAGEN-IMGN_AUDIT_CHILD_FU_RC5-IMAGEN_DIGEST.csv",
        AUDIT_FU2 = DATA_DIR + "FU2/psytools/IMAGEN-IMGN_AUDIT_CHILD_FU2-IMAGEN_DIGEST.csv",
        AUDIT_FU3 = DATA_DIR + "FU3/psytools/IMAGEN-IMGN_AUDIT_FU3.csv",
        ESPAD_BL  = DATA_DIR + "BL/psytools/IMAGEN-IMGN_ESPAD_CHILD_RC5-IMAGEN_DIGEST.csv",
        ESPAD_FU1 = DATA_DIR + "FU1/psytools/IMAGEN-IMGN_ESPAD_CHILD_FU_RC5-IMAGEN_DIGEST.csv",
        ESPAD_FU2 = DATA_DIR + "FU2/psytools/IMAGEN-IMGN_ESPAD_CHILD_FU2-IMAGEN_DIGEST.csv",
        ESPAD_FU3 = DATA_DIR + "FU3/psytools/IMAGEN-IMGN_ESPAD_FU3.csv",
          
        AUDIT_GM       = DATA_DIR + "growthcurves/AUDIT/IMGN_AUDITtotal_GM.csv",
        AUDIT_GM_FINE  = DATA_DIR + "growthcurves/AUDIT/IMGN_AUDITtotal_GM_FINE.csv",
        ESPAD_GM       = DATA_DIR + "growthcurves/ESPAD/IMGN_ESPAD19b_GM.csv",
        ESPAD_GM_FINE  = DATA_DIR + "growthcurves/ESPAD/IMGN_ESPAD19b_GM_FINE.csv",
        
        PHENOTYPE = DATA_DIR + "combinations/drinking_phenotype/Seo_drinking_phenotype_fu2.csv",
        OUR_COMBO = DATA_DIR + "combinations/our_custom_combo/our_custom_combo.csv",
        )

qs_is_raw_csv = dict(
        AUDIT_BL  = True, 
        AUDIT_FU1 = True, 
        AUDIT_FU2 = True, 
        AUDIT_FU3 = True, 
        ESPAD_BL  = True, 
        ESPAD_FU1 = True, 
        ESPAD_FU2 = True, 
        ESPAD_FU3 = True,           
        AUDIT_GM       = False,
        AUDIT_GM_FINE  = False, 
        ESPAD_GM       = False,
        ESPAD_GM_FINE  = False,        
        PHENOTYPE = False, 
        OUR_COMBO = False, 
        )

class Imagen:
        
    def __init__(self, DATA_DIR="/ritter/share/data/IMAGEN", exclude_holdout=True):
        """
        Arguments:
                DATA_DIR: IMAGEN's base directory containing the 'IMAGEN_BIDS' and the 'IMAGEN_RAW' folders

        A dataloader for the IMAGEN data set. 
        Provides following methods on the IMAGEN directory:
            1) Changing read/write permissions on the train set and test set
            2) Analyse different IMAGEN psychometric data:
                a) visualize distribution of values and binarize them for ML classification
                c) visualize distribution of sex and site and counter balance them
            2) Creating HDF5 files for ML experiment pipelines:
                a) Convert imaging data to HDF5 with downsampling option
                b) add sex, site and labels to the HDF5
            3) Plot demographic information of the IMAGEN_BIDS folder
            
        To prepare a hdf5 for ML experiments, the following methods should be called in the specific order:
            1) self.load_label() : prepares a dataframe with subject IDs and a label 'y'
            2) self.prepare_X()  : loads the MRI images or the freesurfer stats as 'X'
            3) self.save_h5()    : saves the 'X'and 'y', along with confounds 'sex' and 'site' as one h5 file 
            for further ML experiments

        This class is an adaptation of Evert's class 'InputCreator' at 
        https://github.com/evertdeman/imagen_thesis/blob/master/inputcreator/inputcreator2.py
        """    
        self.DATA_DIR = DATA_DIR
        self.holdout = exclude_holdout
        
        # load csv files from IMAGEN_BIDS 
        df_BL = pd.read_csv(f"{self.DATA_DIR}/IMAGEN_BIDS/participants_BL.csv", index_col="ID")
        df_FU2 = pd.read_csv(f"{self.DATA_DIR}/IMAGEN_BIDS/participants_FU2.csv", index_col="ID")
        df_FU3 = pd.read_csv(f"{self.DATA_DIR}/IMAGEN_BIDS/participants_FU3.csv", index_col="ID")
        # merge timepoints into one df with hierarchical column levels
        self.df = pd.concat([df_BL, df_FU2, df_FU3], axis=1, keys=["BL","FU2","FU3"])

        if self.holdout:
            # separate the holdout
            self.df_holdout = self.df[self.df["BL"]["holdout"]]
            # and remove the holdout from the main dataframe
            self.df = self.df[~self.df["BL"]["holdout"]]
            
        self.df_out = pd.DataFrame(index=self.df.index)
            
        self.hdf5_name_y = ""
        self.hdf5_name_x = ""
        self.X_colnames = []
        self.all_labels = []
        self.all_confs = []
        
        
    ### PLOTTING FUNCTIONS ###
    def plot_demographics(self, df=None):
        # if no df is explicitly provided then use the entire df 
        if df is None: 
            if self.holdout:
                df = pd.concat([self.df, self.df_holdout])
            else:
                df = self.df
            
        f,axes = plt.subplots(2,2, figsize=(12,8))
        
        plotGraph(df["BL"], "sex", plt_type="pie", ax=axes[0][0]) 
        # plotGraph(df_BL, "gender", plt_type="pie", ax=axes[0][1]) 
        plotGraph(df["BL"], "handedness", plt_type="pie", ax=axes[0][1]) 
#         plotGraph(df["BL"], "site", plt_type="barh", ax=axes[1][0])
        df["BL"][[ "sex", "site"]].value_counts().unstack().T.plot.barh(
            stacked=True, ax=axes[1][0], ylabel='', title="recruitment sites", width=0.9)
        df_paths = df.filter(regex=("path.*")).notnull().sum().unstack()
        df_paths.columns = df_paths.columns.str.replace("path_","")
        df_paths.T.sort_index().plot.barh(
            ax=axes[1][1],title="MRI modalities count", width=0.9)
        
        plt.tight_layout()
#         plt.show()
        
        
    def plot_timepoint_dists(self, 
                             df=None, 
                             cols=["path_T1w", "path_fs-r1_T1w", "path_FLAIR", "path_T2"]):
        # if no df is explicitly provided then use the entire df 
        if df is None: df = self.df 
        
        f,axes = plt.subplots(2,2, figsize=(12,10))
        axes = axes.reshape(-1)

        for i, col in enumerate(cols):
            # drop rows with no imaging data
            df1 = df.swaplevel(axis=1)[col]
            df2 = pd.DataFrame()
            df2["BL  only"] = (~df1["BL"].isnull()) & df1["FU2"].isnull() & df1["FU3"].isnull()
            df2["FU2 only"] = (df1["BL"].isnull()) & (~df1["FU2"].isnull()) & df1["FU3"].isnull()
            df2["FU3 only"] = (df1["BL"].isnull()) & df1["FU2"].isnull() & (~df1["FU3"].isnull())
            df2["BL and FU2"] = (~df1["BL"].isnull()) & (~df1["FU2"].isnull()) & df1["FU3"].isnull()
            df2["BL and FU3"] = (~df1["BL"].isnull()) & (df1["FU2"].isnull()) & (~df1["FU3"].isnull())
            df2["FU2 and FU3"] = (df1["BL"].isnull()) & (~df1["FU2"].isnull()) & (~df1["FU3"].isnull())
            df2["BL and FU2 and FU3"] = (~df1["BL"].isnull()) & (~df1["FU2"].isnull()) & (~df1["FU3"].isnull())

            explode = np.arange(len(df2.columns))*0.1

            df2.sum().sort_index().plot.pie(ax=axes[i],
                    title="{} MRI collected at:".format(col.replace("path_","")), 
                    autopct=lambda p : '{:,.0f}'.format(p*len(df2)/100), 
                    shadow=True, startangle=270, ylabel='', explode=explode)
        plt.tight_layout()
        plt.show()
        
        
    ### DATA SPLIT FUNCTIONS ###
    def _lock_holdout(self):
        assert self.holdout, "holdout was not created when class was initialised"
        holdouts=[]
        for s, row in self.df_holdout["BL"].iterrows():
            if row["holdout"] and pd.notnull(row["subjectID"]):
                holdouts.append(row["subjectID"])
                os.system("chmod 111 {}/IMAGEN_BIDS/{}".format(self.DATA_DIR, row["subjectID"]))
        print("removed read permission for {} subject dirs".format(len(holdouts)))
        
    def _unlock_holdout(self):
        holdouts=[]
        for s, row in self.df_holdout["BL"].iterrows():
            if row["holdout"] and pd.notnull(row["subjectID"]):
                holdouts.append(row["subjectID"])
                os.system("chmod 551 {}/IMAGEN_BIDS/{}".format(self.DATA_DIR, row["subjectID"]))
        print("enabled read permission for {} subject dirs".format(len(holdouts)))
        
        
    ### PREPARE INPUT (X) and OUTPUT (y) IN DF_OUT FOR WRITING TO H5 ###
    def load_label(self, dfq, col, y_colname="q",
                   binarize=False, class0=None, class1=None,
                   viz=True):
        '''load a column from one of the questionnaires and prepare the df_out
            Output:
                Prepares 2 class attributes:
                (1) self.df_out: a pandas dataframe that contains index ["ID"] and columns with 
                the loaded questionnaire or binarized label
                (2) self.hdf5_name_y: The name suffix to add to the final hdf5 file that will be created
                
        If a label is created externally then ensure the above 2 attributes are prepared before proceeding
        to self.prepare_X() and self.save_h5()            
        '''                    
        dfq = dfq.set_index("ID")
        # set NaN values also as class 0: following instructions in page5 of doc: https://imagen-europe.com/wp-content/uploads/sites/234/2020/11/SOP_Annex_IMAGEN_FU3_v8_.pdf
        dfq = dfq.fillna(-1)

        self.df_out.loc[:,y_colname] = dfq.loc[:,col]
        
        # plot distribution of the questionnaire's values
        if viz:
            plotGraph(self.df_out, y_colname)
            plt.show()
    
        # Convert the label to binary if requested
        namesuffix = ""
        if binarize:
            assert (class0 is not None) and (class1 is not None),\
"if 'binarize' is requested then class0 and class1 cases should be specified."
        
            q = self.df_out[y_colname].copy()
            self.df_out[y_colname] = np.nan # reset values to add binary values
            
            if isinstance(class0, int):
                self.df_out.loc[(q<=class0)|(q==-1), y_colname] = 0 # -1 is class 0
                namesuffix += "l{}".format(class0)
            elif isinstance(class0, list):
                self.df_out.loc[q.isin(class0+[-1]), y_colname] = 0 # -1 is class 0
                namesuffix += "l{}".format("".join([str(y) for y in class0]))
            else:
                print("[ERROR] binarizing failed. Incoherent values given for class0=", class0) 
                                           
            if isinstance(class1, int):
                self.df_out.loc[q>=class1, y_colname] = 1
                namesuffix += "u{}".format(class1)
            elif isinstance(class1, list):
                self.df_out.loc[q.isin(class1), y_colname] = 1
                namesuffix += "u{}".format("".join([str(y) for y in class1]))
            else:
                print("[ERROR] binarizing failed. Incoherent values given for class1=", class1)   
                                           
            # plot distribution of the binarized values
            if viz:
                plotGraph(self.df_out, y_colname)
                map_xlabels={'0.0':'Class 0', '1.0':'Class 1', 'nan':'dropped'}
                plt.gca().set_xticklabels([map_xlabels[t.get_text()] for t in plt.gca().get_xticklabels()])
                plt.show()                
        
        if y_colname not in self.all_labels: self.all_labels.extend([y_colname])        
        
        # save the labelname in file name (also append binarizing rules to the final filename)
        if y_colname not in self.hdf5_name_y:
            self.hdf5_name_y += y_colname + namesuffix
        
        return self.df_out
            
            
            
    def prepare_X(self, tp_scan, 
                  preloaded_X=None, mri_col="", feature_cols=".+",  
                  confs=["sex", "site"], 
                  viz=True):
        '''Prepare the "X" column in the self.df_out (that gets converted to hdf5).
        X can be set as either,
        (a) one of MRI modalities (by specifing a "mri_col" present in the 'participants_*.csv)'
        (b) the freesurfer extracted stats like grey matter thickness or volume or DTI features 
            (by specifing a list of "feature_cols" present in the 'sMRI-derivatives_*.csv)
        (c) or a preloaded numpy array or a pandas dataframe saved as numpy array
        
        Args:
            feature_cols: regex pattern specifing which columns to include from self.get_feature_colnames()
        '''       
        
        assert self.df_out is not None, "first assign a label from the questionnaires for the hdf5 conversion.\nUse the methods self.load_label()"
        for c in confs:
            if c not in self.all_confs:
                self.all_confs.extend([c])
        # first, add sex and site columns to df_out as they are confounds
        self.df_out[confs] = self.df[tp_scan][confs]
        
        # if a preloaded MRI data is provided directly as a dict of {subjectID: loaded_data_array}
        if preloaded_X is not None and (
        isinstance(preloaded_X, (pd.DataFrame, pd.Series))):
            self.X_colnames = list(preloaded_X.columns)
            self.df_out = self.df_out.merge(preloaded_X, on="ID")      
            # name of output hdf5
            self.hdf5_name_x += "-{}".format(tp_scan) 
        # if a MRI column name from BIDS csv file is provided then set this path as X 
        elif mri_col: 
            self.X_colnames = ["path"]
            self.df_out["path"] = self.df[tp_scan][mri_col]
            # name of output hdf5
            self.hdf5_name_x = "{}-{}".format(
                mri_col.replace("path_","").replace("_","").replace("-",""), tp_scan)
        else:        
            feature_filtername ="all" if feature_cols==".+" else feature_cols.replace(".","")
            self.hdf5_name_x = "sMRIfeatures-{}-{}".format(feature_filtername, tp_scan)             
            feature = pd.read_csv(
                join(self.DATA_DIR, f"IMAGEN_BIDS/sMRI-derivatives_{tp_scan}.csv"),
                index_col="ID")
            # filter columns
            feature = feature.filter(regex=feature_cols)
            self.X_colnames = list(feature.columns)
            self.df_out = self.df_out.merge(feature, on="ID")
            
        if viz: print(self.df_out[self.all_labels[0]].map(
            {0: 'Safe users', 1: 'Heavy misusers', np.nan: 'Moderate misusers'}
        ).value_counts().sort_index(ascending=False))  
            
        # only retain rows that have both X data and label data available        
        self.df_out = self.df_out.dropna()
        print(f"Final dataframe prepared. \nTotal subjects = {len(self.df_out)}")
        # plot the final distribution of labels and confounds
        if viz: 
            self.final_df_plot()
            
        # map confounds to dummy encoding 
        for c in self.all_confs:
            self.df_out[c] = self.df_out[c].astype('category').cat.codes.astype(int)
        
            
### SAVE TO HDF5 ###
    def save_h5(self, mri_kwargs={"z_factor":1}):
        
        # determine the final h5 filename
        # if mri downsample is requested, then add this info in the final hdf5 filename
        z_info = "-z{:1.0f}".format(1/mri_kwargs["z_factor"])
        if mri_kwargs["z_factor"] != 1 and z_info not in self.hdf5_name_x:
            self.hdf5_name_x += z_info
            
        filename = "{}_{}_n{}".format(self.hdf5_name_x, 
                                        self.hdf5_name_y,
                                        len(self.df_out))
        dest = join(self.DATA_DIR, "h5files", slugify(filename)+".h5")      
        # if file already exists, then print error and exit
        if os.path.isfile(dest): 
            print(f"hdf5 file already exists at {dest}. First delete it manually.")
            return
        else:
            print(f"saving h5 file at {dest}")        
        
        X = self.df_out[self.X_colnames].to_numpy()
        # if MRI paths are provided as 'X' then load these MRI images into a np array
        if isinstance(X[0,0], str):
            mri_paths = list(self.df_out[self.X_colnames[0]])
            X, mri_idxs = self._load_mri(mri_paths, **mri_kwargs)
            self.X_colnames = mri_idxs        
        else: #isinstance(X[0,0],(np.ndarray, np.generic)):
            X = np.stack(X)
            
        # write all data to a hdf5 file
        h5=h5py.File(dest, "w")
        h5.create_dataset('X', data=X, chunks=True)
        h5.create_dataset('i', data=self.df_out.index)
        for y in self.all_labels:
            h5.create_dataset(y, data=self.df_out[y])
        for c in self.all_confs:
            h5.create_dataset(c, data=self.df_out[c])
        # set attributes to distinguish labels from confounds
        h5.attrs['labels']= self.all_labels
        h5.attrs['confs']= self.all_confs
        # set attribute to name the X features for later interpretations
        h5.attrs['X_col_names']= self.X_colnames
        h5.close()
    
    def _load_mri(self, paths, 
                  z_factor=1, apply_mask=None, z_order=3, z_prefilter=True,
                  variance_threshold=False):
        # initialize the final matrix      
        data_idxs = np.arange(len(paths))
        print("Extracting {} images into a single matrix..".format(len(data_idxs)))  
        data = []
        for i, path in tqdm(enumerate(paths), total=len(paths)):
            scan = nib.load(path)
            img_arr = scan.get_fdata().astype(np.float32)
            # interpolate (zoom) to a smaller size and remove decimals created from the interpolation
            if z_factor != 1:
                img_arr = zoom(img_arr, z_factor, 
                               order=z_order, 
                               prefilter=z_prefilter) 
                data_idxs = data_idxs # todo update the idxs
                # round off to remove the artifacts created in empty regions by zoom()
                if z_order>1 and np.mean(img_arr)>1:
                    img_arr = np.around(img_arr, 0)
#                 plt.imshow(img_arr[:, :, 40])

            if apply_mask is not None:
                img_arr = apply_mask(img_arr)
                data_idxs = apply_mask(data_idxs)                
                
            data.extend([img_arr])
            
        data_matrix = np.stack(data)
#         plt.hist(data_matrix.ravel(), bins=100)
        return data_matrix, data_idxs
    
    
    def get_features_colnames(self, tp="BL"):
        return list(pd.read_csv(join(self.DATA_DIR, f"IMAGEN_BIDS/sMRI-derivatives_{tp}.csv")).columns)

    def get_mri_colnames(self, tp="BL"):
        return list(pd.read_csv(join(self.DATA_DIR, f"IMAGEN_BIDS/participants_{tp}.csv")).columns)
    
    def final_df_plot(self):
        print("Distributions in final dataframe:")            
        cols = self.all_confs + self.all_labels
        f, axes = plt.subplots(1,len(cols), figsize=(5*len(cols),4), sharey=True)
        for i, col in enumerate(cols):                
            self.df_out[col].value_counts().sort_index().plot.bar(ax=axes[i], title=col)
        plt.show()
        print(f"Total subjects = {len(self.df_out)}")
        
        
####################################################################################################################
def create_h5s(lbl_combos, name, x_tp="FU3", data_subset_to_use='trainval', viz=0, feature_cols=".+"):
    
    for csv, col, c0, c1, colname in lbl_combos:
        if data_subset_to_use=='all':
            d = Imagen(exclude_holdout=False)
        elif data_subset_to_use=='holdout':  
            d = Imagen()
            d.df = d.df_holdout
            d.df_out = pd.DataFrame(index=d.df_holdout.index)
        else:
            d = Imagen(exclude_holdout=True)         

        if qs_is_raw_csv[csv]:
            dfq = pd.read_csv(qs[csv], usecols=["User code", col], dtype={"User code":str})
            dfq["ID"] = dfq["User code"].str.replace("-C", "").replace("-I", "").astype(int)
            dfq = dfq.drop("User code", axis=1)
        else:
            dfq = pd.read_csv(qs[csv], usecols=["ID", col])
        
        d.load_label(dfq, col=col, viz=(viz>1), binarize=True, class0=c0, class1=c1, y_colname=colname)
        d.prepare_X(x_tp, feature_cols=feature_cols, viz=(viz>0))
        
        d.hdf5_name_x = name
        d.hdf5_name_y = csv + "-" + col + "-" + colname + "-"
        if viz: 
            plt.show()
            print("shape of X", d.df_out[d.X_colnames].values.shape)
        else: 
            d.save_h5()
            
def print_h5list(fil):
    for f in sorted(glob(f"/ritter/share/data/IMAGEN/h5files/*{fil}*.h5")):
        print(f" '{f}',")
        
        
####################################################################################################################
# clustering labels 
from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from matplotlib import cm

def determine_clusters_k(scaled_features, 
                         kmeans_kwargs = {"init":"random",
                                         "n_init": 10,
                                         "max_iter": 300,
                                         "random_state": 42}):
    ''' determine the most appropriate number of clusters'''
    sse = []
    silhouette_coefficients = []
    max_k =  12
    for k in range(1,max_k+1):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(scaled_features)
        sse.append(kmeans.inertia_)
        if k>1:
            score = silhouette_score(scaled_features, kmeans.labels_)
            silhouette_coefficients.append(score)

    f, axes = plt.subplots(1, 2, figsize=(12,5), sharex=True)
    plt.style.use("fivethirtyeight")
    # plot elbow
    axes[0].plot(range(1,max_k+1), sse)
    axes[0].set_xticks(range(1, max_k+1))
    axes[0].set_xlabel("Number of Clusters")
    axes[0].set_ylabel("SSE")

    # plot silhouette
    axes[1].plot(range(2,max_k+1), silhouette_coefficients)
    axes[1].set_xlabel("Number of Clusters")
    axes[1].set_ylabel("Silhouette Coefficient")

    plt.show()
    
def viz_clusters(df):
    
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12,5))
    clusters = df.cluster.unique()
    colors = cm.get_cmap('Dark2', lut=len(clusters))
    
    # calculate the mean and std dev of the 3 clusters
    x = np.arange(-1, 1.25, step=0.25)
    
    for ax, typ in zip(axes, ["", "_scaled"]):       

        for c, dfi in df.groupby("cluster"):            
            # y = c + m*x (with numpy broadcast)
            yi = dfi[f"Intercept{typ}"].values.reshape(-1,1) + (dfi[f"Slope{typ}"].values.reshape(-1,1) * x.reshape(1,-1))
            ax.plot(x, yi.mean(axis=0), lw=2, label=f'cluster {c} mean', color=colors(c))
            ax.fill_between(x, yi.mean(axis=0) + yi.std(axis=0), yi.mean(axis=0) - yi.std(axis=0), 
                            facecolor=colors(c), alpha=0.5)
        ax.set_title("clusters ({})".format(typ.replace("_","")))
        
    ax.legend(loc='upper left')
    plt.show()
    
    # print counts per cluster
    print("Per cluster count: \n{}".format(df.cluster.value_counts().sort_index()))
    
    
def abline(ax, slope, intercept, color=None):
    """Plot a line from slope and intercept"""
    x_vals = np.arange(-1, 1.25, step=0.25)
    y_vals = intercept + slope*x_vals
    ax.set_ylim(-3, 3)
    
    if color is None:
        color = plt.get_cmap('coolwarm')((intercept+slope))
        
    ax.plot(x_vals, y_vals, color=color, alpha=0.1
            , scalex=False, scaley=False, linewidth=2)
    