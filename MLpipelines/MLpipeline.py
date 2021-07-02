from os.path import join
import h5py
import numpy as np
import pandas as pd
from datetime import datetime 
import statsmodels.api as sm
from joblib import Parallel, delayed

from functools import partial
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import balanced_accuracy_score, accuracy_score, roc_auc_score, get_scorer, make_scorer, confusion_matrix
from sklearn.model_selection import train_test_split, permutation_test_score, KFold, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import estimator_html_repr
from sklearn.base import clone
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from scipy.stats import chi2_contingency, chi2


class MLpipeline:
    
    def __init__(self, parallelize=True, random_state=None, debug=False):
        """ Define parameters for the MLLearn object.
        Args:
            "h5file" (str): Full path to the HDF5 file.
            "n_jobs" (int): Number of cores to use in parallel. 
            "conf_list" (list of str): the list of confound names to load from the hdf5.
            random_state (int): A random state to get reproducible results
        """        
        self.random_state = random_state
        self.debug = debug
        # donot parallelize in debug mode
        self.parallelize = parallelize if not self.debug else False 
        self.confs = {}  
        
        self.X_test = None
        self.y_test = None
        self.confs_test = {} 
        
        # These are defined to calculate both BA and AUC later.
        self.func_acc = get_scorer('accuracy')
        self.func_bal_acc = make_scorer(balanced_accuracy_score) #, adjusted=True
        self.func_auc = get_scorer("roc_auc")
        

    def load_data(self, f, X="X", y="y", confs=[], group_confs=False):
        """ Load the data from the HDF5 file, 
            The neuroimaging data/ input data is saved under "X", the label under "y" and 
            the confounds from the dict self.confs
            If group_confs==True, groups all confounds into one numpy array and saves in
            self.confs["group"]
        """
        if isinstance(f, dict):
            df = f
        else:
            df = h5py.File(f, "r")
        self.X = np.array(df[X])
        self.y = np.array(df[y])
        self.sub_ids = np.array(df["i"])
        
        # self.X needs to be flattened as sklearn expects 2D input.
        if self.X.ndim > 2:
            self.X = self.X.reshape([self.X.shape[0], np.prod(self.X[0].shape)])
        elif self.X.ndim == 1:
            self.X = self.X.reshape(-1,1) # sklearn expects 2D input
            
        # Load the confounding variables into a dictionary, if there are any
        for c in confs:
            if c in df.keys():
                v = np.array(df[c]) #todo bugfix: assumed that all confounds are categorical
                self.confs[c] = v 
                
                if group_confs:
                    if "group" not in self.confs:
                        self.confs["group"] = v
                    else:
                        self.confs["group"] = v + 100*self.confs["group"]
                        #todo bugfix assumption that confounds have values in range [0,100]    
            else:
                print(f"[WARN]: confound {c} not found in hdf5 file.")          
        

    def train_test_split(self, test_idx=[], test_size=0.25, stratify_by_conf=None):
        """ Split the data into a test set (_test) and a train+val set.

        Args:
            stratify_group (str, optional): A string that indicates a confound name. 
                Get confounds list available from the dict self.confs.
                If a confound is selected, subjects will be stratified according 
                to confound and outcome label. Defaults to None, in which case 
                subjects are only stratified according to the outcome label.
        """
        # if test_idx are already provided then dont generate the test_idx yourself
        if not len(test_idx): 
            # by default, always stratify by label
            stratify = self.y
            if stratify_by_conf is not None:
                stratify = self.y + 100*self.confs[stratify_by_conf]
                #todo bugfix assumption that confounds have values in range [0,100]          

            _, test_idx = train_test_split(range(len(self.X)), 
                                                      test_size=test_size, 
                                                      stratify=stratify, 
                                                      shuffle=True, 
                                                      random_state=self.random_state)
            
        test_mask = np.zeros(len(self.X), dtype=bool)
        test_mask[test_idx] = True
            
        self.X_test = self.X[test_mask]
        self.y_test = self.y[test_mask]
        self.sub_ids_test = self.sub_ids[test_mask]
        self.X = self.X[~test_mask]
        self.y = self.y[~test_mask]
        self.sub_ids = self.sub_ids[~test_mask]
        
        for c in self.confs:
            self.confs_test[c] = self.confs[c][test_mask]
        for c in self.confs:
            self.confs[c] = self.confs[c][~test_mask]
        
        self.n_samples_tv = len(self.y)
        self.n_samples_test = len(self.y_test)

        
    def transform_data(self, func):
        '''performs func.transform() (see sklearn structures)
                on either (a) all the data loaded, if performed before train_test_split()
                    or on (b) the train+val subset, if performed after train_test_split()'''
        # first check if train_test_split has already been performed
        if self.X is not None:
            self.X = func.transform(self.X)
            self.y = func.transform(self.y)

            for c in self.confs:
                self.confs[c] = func.transform(self.confs[c])
                
        
    def change_input_to_conf(self, c, onehot=True):
        """ Change the inputs of the model to a vector containing a confound.

        Args:
            c (str): The name of the confound from the list loaded in conf_dict. 
                            Get confounds list available from the dict self.confs.
            onehot (bool, optional): Whether to one-hot encode the new input. This should be 
                enabled for linear/logistic regression models. Defaults to True.
        """
        assert self.y_test is not None, "self.train_test_split() should be run before calling self.change_input_to()"
        self.X = self.confs[c].reshape(-1, 1)
        self.X_test = self.confs_test[c].reshape(-1, 1)
        
        if onehot:
            self.X = OneHotEncoder(sparse=False).fit_transform(self.X)
            self.X_test = OneHotEncoder(sparse=False).fit_transform(self.X_test)


    def change_output_to_conf(self, c):
        """ Change the targets (outputs) of the classifier to a confound vector.

        Args:
            c (str): The name of the confound which is loaded in self.confs, 
                Get confounds list using self.confs().
        """        
        assert self.y_test is not None, "self.train_test_split() should be run before calling self.change_output_to()"
        self.y = self.confs[c]
        self.y_test = self.confs_test[c]
        
            
    @ignore_warnings(category=ConvergenceWarning)
    def run(self, pipe, grid, 
            n_splits=5, stratify_by_conf=None, conf_corr_params={}, 
            run_pbcc=False, permute=0):
        """ The main function to run the classification. 

        Args:
            pipe (sklearn Pipeline): a pipeline containing preprocessing steps 
                and the classification model. 
            grid (dict): A dict of hyperparameter lists that will be passed to 
                sklearn's GridSearchCV() function as 'param_grid'
            permute (int, optional): Number of permutations to perform. Defaults to 0 i.e.
                no permutations are performed.

        Returns:
            results (dict): A dictionary containing classification metrics for the 
                best parameters.
            best_params (dict): The best parameters found by grid search.
        """        
        assert self.X_test is not None, "self.train_test_split() has not been run yet. \
First split the data into train and test data"
        
        # create the inner cv folds on the trainval split
        cv = StratifiedKFold(n_splits, shuffle=True, random_state=self.random_state)
        # stratify the inner cv folds by the label and confound groups
        stratify = self.y
        if stratify_by_conf is not None:
            stratify = self.y + 100*self.confs[stratify_by_conf]            
        cv_splits = cv.split(self.X, stratify)
        
        n_jobs = n_splits if (self.parallelize) else None 
        # grid search for hyperparameter tuning with the inner cv         
        gs = GridSearchCV(estimator=pipe, param_grid=grid, n_jobs=n_jobs,
                          cv=cv_splits, scoring=self.func_bal_acc,
                          return_train_score=True, refit=True, verbose=2)
        # fit the estimator on data
        gs = gs.fit(self.X, self.y, **conf_corr_params)
        
        self.estimator = gs.best_estimator_
        
        # store balanced_accuracy scores
        train_score = np.mean(gs.cv_results_["mean_train_score"])
        valid_score = gs.best_score_ # mean cross-validated score of the best_estimator
        test_score = gs.score(self.X_test, self.y_test)
        
        # Calculate AUC if label is binary
        roc_auc = np.nan        
        if len(np.unique(self.y_test))==2:
            roc_auc = self.func_auc(self.estimator, self.X_test, self.y_test)       
        
        # save the predicted probability scores on the test subjects (for trying other metrics later)
        test_probs = np.around(gs.predict_proba(self.X_test), decimals=4)
        test_lbls = self.y_test
        test_ids = self.sub_ids_test
        
        results_pbcc = {}
        if run_pbcc:
            r_pbcc = self.run_pbcc(gs.best_estimator_, self.X_test, self.y_test, self.confs_test)
            results_pbcc = {
                "d2_conf": r_pbcc[0],
                "d2_pred": r_pbcc[1],
                "d2_both": r_pbcc[2],
                }
            
        # if permutation is requested, then calculate the test statistic after permuting y  
        results_pt = {}        
        if permute:
            # run parallel on all free CPUs except one
            n_jobs = -2 if (self.parallelize) else 1
            
            with Parallel(n_jobs=n_jobs) as parallel:
                # run parallel jobs on all cores at once
                pt_scores = parallel(delayed(
                                MLpipeline._one_permutation)(
                                    self.X, self.y, self.X_test, self.y_test,
                                    pipe, grid, fit_params=conf_corr_params, run_pbcc=run_pbcc, confs_test=self.confs_test,
                                    n_splits=n_splits, score_func=self.func_bal_acc, score_func_auc=self.func_auc,
#                                     permute_y=False, permute_x=True, permute_y_test=False, permute_x_test=False         
                )  for _ in range(permute))
            
            pt_scores  = np.array(pt_scores)           
            results_pt = {"permuted_test_score" : pt_scores[:,0].tolist()}
            
            if not np.isnan(pt_scores[:,1]).all():
                results_pt.update({"permuted_roc_auc": pt_scores[:,1].tolist()})
                
            if run_pbcc:
                results_pt.update({
#                 "permuted_d2_conf": pt_scores[:,2].tolist(), # will always be same
                "permuted_d2_pred": pt_scores[:,3].tolist(),
                "permuted_d2_both": pt_scores[:,4].tolist()
                })   
            
        results = {
            "train_score" : train_score, 
            "valid_score" : valid_score, 
            "test_score" : test_score,
            "roc_auc" : roc_auc, 
            "test_ids" : test_ids.tolist(),
            "test_lbls" : test_lbls.tolist(),
            "test_probs" : test_probs.tolist(),
            **results_pbcc,
            **results_pt,
            **gs.best_params_,
            }
        
        return results
    

    @staticmethod
    def sensitivity_score(y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tp/(tp+fn)

    
    @staticmethod
    def specificity_score(y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn/(tn+fp) 
    
    
    @staticmethod
    def _shuffle(arr, groups=None):
        """ 
        Permute a vector or array y along axis 0.
        Args:
            arr (numpy array): The to-be-permuted array. The array will be permuted
                along axis 0.  
            groups (str): If set, the permutations of y will only happen within members
                of the same group. Get confounds list available from the dict self.confs
        Returns:
            y_permuted: The permuted array y_permuted. """    
        if groups is None:
            indices = np.random.permutation(len(arr))
        else:
            indices = np.arange(len(groups))
            for group in np.unique(groups):
                this_mask = (groups == group)
                indices[this_mask] = np.random.permutation(indices[this_mask])
                
        return arr[indices]
    
    @staticmethod
    def _one_permutation(X, y, X_test, y_test, 
                        estimator, grid, fit_params={}, run_pbcc=False, confs_test=None,
                        n_splits=5, score_func=None, score_func_auc=get_scorer("roc_auc"),
                        permute_test=False):

        """ Run the standard gridsearch+cv pipeline once after permuting X or y
        and evaluate on the test set X_test, y_test
        Args:

        Returns:
            pt_score (float): Balanced accuracy score from permuted samples.
            pt_score_auc (float): ROC AUC score from permuted samples.
            d2_* (float): D2 scores for fitting the PBCC models with different
                independent variables *. See function pbcc().
        """
        # By default X is shuffled rather than the labels y as the relationship 
        # between confound and label should be maintained for PBCC. refer Dinga et al., 2020.
        X = MLpipeline._shuffle(X)
            
        if permute_test: 
            X_test = MLpipeline._shuffle(X_test) 
            
        # create the inner crossvalidation folds on the trainval split
        # random state is not fixed because each permutation must be completely random
        cv = StratifiedKFold(n_splits, shuffle=True, random_state=None) 
        # grid search for hyperparameter tuning with the inner cv 
        gs = GridSearchCV(estimator, param_grid=grid, n_jobs=None,
                          cv=cv, scoring=score_func,
                          return_train_score=False, refit=True, verbose=0)
        
        # disable the random_state in the confound correction to get varying scores
        if "conf_corr_cb__cb_by" in fit_params:
            estimator["conf_corr_cb"].random_state = None
            
        # fit the estimator on data            
        gs = gs.fit(X, y, **fit_params)

        pt_score = gs.score(X_test, y_test)
        
        # calc auc score
        pt_score_auc = np.nan
        if len(np.unique(y)) == 2:
            pt_score_auc = score_func_auc(gs.best_estimator_, X_test, y_test)

        # calc PBCC scores
        pbcc_results = []
        if run_pbcc:
            pbcc_results = MLpipeline.run_pbcc(gs.best_estimator_, X_test, y_test, confs_test)
            

        return [pt_score, pt_score_auc, *pbcc_results]
    

    @staticmethod
    # other statistical tests for influence of confounds
    def run_pbcc(estimator, X_test, y_test, confs_test):
        """ Calculate D2 scores for the Prediction-Based Confound Control by Dinga
            et al., 2020. 
        Returns:
        """    
        
        c = []
        # todo: only works for categorical confounded
        for ci in confs_test:
            if ci != 'group':
                ci_vals = confs_test[ci]
                # Multiclass categorical confounds (such as site) should be OneHotEncoded to avoid
                # any falsely assumed ordinal relationship between categories
                if max(ci_vals)>1: 
                    ci_vals = OneHotEncoder(sparse=False).fit_transform(ci_vals.reshape(-1,1))
                c.extend([ci_vals])                
        c = np.column_stack(c)
        
        p = estimator.predict(X_test)
        
        # if it is not a binary classification, use a multinomial logit model
        if max(y_test)>1:
            logit = sm.MNLogit
            p = OneHotEncoder(sparse=False).fit_transform(p.reshape(-1,1))
        else:
            logit = sm.Logit
            
        pc = np.column_stack([p, c])
        
        # Add intercepts
        p = sm.add_constant(p, prepend=False)
        c = sm.add_constant(c, prepend=False)
        pc = sm.add_constant(pc, prepend=False)
        
        # Fit models with different independent variables and extract D^2 or squared Deviance
        d2_pred = logit(y_test, p ).fit(method='bfgs').prsquared 
        d2_conf = logit(y_test, c ).fit(method='bfgs').prsquared 
        d2_both = logit(y_test, pc).fit(method='bfgs').prsquared         
        
        return d2_conf, d2_pred, d2_both


def run_chi_sq(data, labels, confs):

    df = pd.DataFrame()
    
    for y in labels:
        for c in confs:
            chi, p, dof, _ = chi2_contingency(pd.crosstab(data[y], data[c]))
            result = {"y":y, "c":c, "chi":chi, "p-value":p, "dof":dof}
            df = df.append(result, ignore_index=True)
    return df