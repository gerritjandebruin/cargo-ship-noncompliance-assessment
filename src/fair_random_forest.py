from copy import deepcopy as copy
import json
import logging
from math import ceil
import multiprocessing
import os
import random
import time
import typing
import warnings

import joblib
from joblib import delayed, Parallel
import numpy as np
from numpy.core.numeric import outer
import pandas as pd
from scipy.stats import mode, entropy
import sklearn.feature_selection
import sklearn.metrics
from sklearn.metrics import roc_auc_score
import sklearn.model_selection
import typer
from tqdm.auto import tqdm

from .logger import logger

iterations = [
    (theta, md+1, n_bins, outer_fold, inner_fold)
    for theta in [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.]
    for md in range(10)
    for n_bins in (2,10)
    for outer_fold in range(5)
    for inner_fold in range(5)
]   


class FairDecisionTreeClassifier():
    """A fair decision tree classifier.
    
    This decision tree classifier enables to optimize both with regards to y and
    b. In the case of orthogonality = 0, fully optimise for y (making it an 
    ordinary decision tree classifier). In the case of orthogonality=1, complete
    make the output of the learned classifier independent towards the bias 
    group.
    
    Parameters:
    - n_bins: Number of bins used to discretise continuous variables.
    - min_leaf: The minimum number of samples required to be in each leaf.
    - max_depth: The maximum depth of each tree.
    - n_samples: The number (int) or fraction (float) of samples to bootstrap.
    - criterion: Either "entropy" or "auc_sub" score criterion for splitting.
    - max_features: Number (int) or proportion (float) of features to bootstrap.
        Alternatively, use "auto" or "sqrt" to take the square root of the total
        number of features to bootstrap, or "log" or "log2" to use the log2 of 
        the total number of features.
    - bias_method: OvR approach for multi-categorical bias attribute, can be any
        of "avg", "w_avg", or "xtr".
    - orthogonality: Float between 0-1 (both endpoints inclusive), see 
        description.
    - bootstrap: Whether to bootstrap with replacement (True) or sample without
        replacement (False).
    - random_state
    - compount_bias_method: Aggregation approach for multiple bias attributes. 
        Can be any of "avg", "xtr".
    """
    def __init__(self,
        n_bins=2, min_leaf=1, max_depth=2, n_samples=1.0, max_features="auto", 
        bootstrap=True, random_state=42, criterion="auc_sub", bias_method="avg", 
        compound_bias_method="avg", orthogonality=.5
        ):
        self.is_fit = False
        self.n_bins = n_bins
        self.min_leaf = min_leaf
        self.max_depth = max_depth
        self.n_samples = n_samples
        self.criterion = criterion
        self.max_features = max_features
        self.bias_method = bias_method
        self.orthogonality = orthogonality
        self.bootstrap = bootstrap
        self.random_state = random_state        
        self.compound_bias_method = compound_bias_method
                                
    def fit(self, X="X", y="y", b="bias"):
        """
        X -> any_dim pandas.df or np.array: numerical/categorical
        y -> one_dim pandas.df or np.array: only binary
        b -> any_dim pandas.df or np.array: treated as str
        """
        np.random.seed(self.random_state)
        self.X = np.array(X)
    
        self.y = np.array(y).astype(int)
        self.b = np.array(b).astype(int)
    
        if (self.X.shape[0]!=self.y.shape[0]) or (self.X.shape[0]!=self.b.shape[0]) or (self.y.shape[0]!=self.b.shape[0]):
            raise Exception("X, y, and b lenghts do not match")    
        if len(self.y.shape)==1 or len(self.y.ravel())==len(self.X):
            self.y = self.y.ravel()
        if len(self.b.shape)==1 or len(self.b.ravel())==len(self.X):
            self.b = self.b.ravel()    
        
        self.b_neg, self.b_pos = 0, 1
        self.y_neg, self.y_pos = 0, 1
        all_indexs = range(X.shape[0])
        all_features = range(X.shape[1])
        self.features = all_features
        # self.samples -> set of indexs according to sampling
        if "int" in str(type(self.n_samples)):
            self.samples = np.array(
                np.random.choice(
                    all_indexs,
                    size=self.n_samples,
                    replace=self.bootstrap
                )
            )
        else:
            self.samples = np.array(
                np.random.choice(
                    all_indexs,
                    size=int(self.n_samples*len(all_indexs)),
                    replace=self.bootstrap,
                )
            )
        
        self.pred_th = sum(self.y[self.samples]==self.y_pos) / len(self.samples)

        def choose_features():   
            if "int" in str(type(self.max_features)):
                chosen_features = np.random.choice(
                        self.features,
                        size=max(1, self.max_features),
                        replace=False
                )
            elif ("auto" in str(self.max_features)) or ("sqrt" in str(self.max_features)):
                chosen_features = np.random.choice(
                        self.features,
                        size=max(1, int(np.sqrt(len(self.features)))),
                        replace=False
                )
            elif "log" in str(self.max_features):
                chosen_features = np.random.choice(
                        self.features,
                        size=max(1, int(np.log2(len(self.features)))),
                        replace=False
                )
            else:
                chosen_features = np.random.choice(
                        self.features,
                        size=max(1, int(self.max_features*len(self.features))),
                        replace=False,
                )
            return chosen_features
    
        # returns a dictionary as {feature: cutoff_candidate_i} meant as <
        def get_candidate_splits(indexs):
            
            candidate_splits = {}
            chosen_features = choose_features()
            #print(chosen_features)
            for feature in chosen_features:
                if "str" in str(type(self.X[0,feature])):
                    candidate_splits[feature] = list(pd.value_counts(self.X[indexs, feature]).keys())
                else:
                    n_unique = len(np.unique(self.X[indexs,feature])) 
                    values = np.unique(self.X[indexs, feature])
                    n_unique = len(values)
                    if (n_unique) > self.n_bins:
                        lo = 1/self.n_bins
                        hi = lo * (self.n_bins-1)
                        quantiles = np.linspace(lo, hi, self.n_bins-1)
                        values = list(np.quantile(values, q=quantiles))
                    candidate_splits[feature] = values

            return candidate_splits

        # return score of split (dependant on criterion) ONLY AUC implemented so far
        def evaluate_split(feature, split_value, indexs):
            
            # get auc of y associatated with split
            def get_auc_y(index_left, index_right):
                
                n_left = len(index_left)
                n_right = len(index_right)
                y_left = self.y[index_left]
                y_right = self.y[index_right]
                proba_left = sum(y_left==1)/n_left
                proba_right = sum(y_right==1)/n_right
                
                y_prob = np.concatenate(
                    (np.repeat(proba_left, n_left), np.repeat(proba_right, n_right))
                )
                y_true = np.concatenate(
                    (y_left, y_right)
                )
            
                auc_y = roc_auc_score(y_true, y_prob)
                
                return auc_y
            
            # get auc of b associatated with split
            def get_auc_b(index_left, index_right):
                indexs = np.concatenate((index_left, index_right))
                if len(self.b.shape)==1: #if we have only 1 bias column
                    b_unique = np.unique(self.b[indexs])
                    
                    n_left = len(index_left)
                    n_right = len(index_right)
                    y_left = self.y[index_left]
                    y_right = self.y[index_right]
                    proba_left = sum(y_left==1)/n_left
                    proba_right = sum(y_right==1)/n_right

                    y_prob = np.concatenate(
                        (np.repeat(proba_left, n_left), np.repeat(proba_right, n_right))
                    )

                    b_left = self.b[index_left]
                    b_right = self.b[index_right]
                    
                    if len(b_unique)==1: #if these indexs only contain 1 bias_value
                        auc_b = 1
                        
                    elif len(b_unique)==2: # if we are dealing with a binary case
                        n_left = len(index_left)
                        n_right = len(index_right)
                        y_left = self.y[index_left]
                        y_right = self.y[index_right]
                        proba_left = sum(y_left==1)/n_left
                        proba_right = sum(y_right==1)/n_right

                        y_prob = np.concatenate(
                            (np.repeat(proba_left, n_left), np.repeat(proba_right, n_right))
                        )
                        
                        b_left = self.b[index_left]
                        b_right = self.b[index_right]
                        b_true = np.concatenate(
                            (b_left, b_right)
                        )
                        auc_b = roc_auc_score(b_true, y_prob)
                        auc_b = max(1-auc_b, auc_b)
                    else: # apply OvR
                        auc_storage = []
                        wts_storage = []
                        for b_uni in b_unique:
                            b_true = np.concatenate(
                                ((b_left==b_uni).astype(int), (b_right==b_uni).astype(int))
                            )
                            auc_b_uni = roc_auc_score(b_true, y_prob)
                            auc_b_uni = max(1-auc_b_uni, auc_b_uni)
                            if np.isnan(auc_b_uni):
                                auc_b_uni = 1
                            auc_storage.append(auc_b_uni)
                            wts_storage.append(sum(b_true))
                        if self.bias_method=="avg":
                            auc_b = np.mean(auc_storage)
                        elif self.bias_method=="w_avg":
                            auc_b = np.average(auc_storage, weights=wts_storage)
                        elif self.bias_method=="xtr":
                            auc_b = max(auc_storage)
                            
                # if we have more than 1 bias column
                else:
                    auc_b_columns = []
                    for b_column in range(self.b.shape[1]):
                        b_unique = np.unique(self.b[indexs, b_column])
                    
                        if len(b_unique)==1: #if these indexs only contain 1 bias_value
                            auc_b = 1

                        elif len(b_unique)==2: # if we are dealing with a binary case
                            true_pos = sum(self.b[index_left, b_column]==b_unique[0])
                            false_pos = sum(self.b[index_left, b_column]==b_unique[1])
                            actual_pos = sum(self.b[indexs, b_column]==b_unique[0])
                            actual_neg = sum(self.b[indexs, b_column]==b_unique[1])
                            tpr = true_pos / actual_pos
                            fpr = false_pos / actual_neg
                            auc_b = (1 + tpr - fpr) / 2
                            auc_b = max(1 - auc_b, auc_b)
                            if np.isnan(auc_b):
                                auc_b = 1
                                
                        else: # apply OvR
                            auc_storage = []
                            wts_storage = []
                            for b_uni in b_unique:
                                true_pos = sum(self.b[index_left, b_column]==b_uni)
                                false_pos = sum(self.b[index_left, b_column]!=b_uni)
                                actual_pos = sum(self.b[indexs, b_column]==b_uni)
                                actual_neg = sum(self.b[indexs, b_column]!=b_uni)
                                tpr = true_pos / actual_pos
                                fpr = false_pos / actual_neg
                                auc_b_uni = (1 + tpr - fpr) / 2
                                auc_b_uni = max(1 - auc_b_uni, auc_b_uni)
                                if np.isnan(auc_b_uni):
                                    auc_b_uni = 1
                                auc_storage.append(auc_b_uni)
                                wts_storage.append(actual_pos)
                            if self.bias_method=="avg":
                                auc_b = np.mean(auc_storage)
                            elif self.bias_method=="w_avg":
                                auc_b = np.average(auc_storage, weights=wts_storage)
                            elif self.bias_method=="xtr":
                                auc_b = max(auc_storage)
                        auc_b_columns.append(auc_b) 
                    if self.compound_bias_method=="avg":
                        auc_b = np.mean(auc_b_columns)
                    elif self.compound_bias_method=="xtr":
                        auc_b = max(auc_b_columns)
                return auc_b 
            
            if "str" in str(type(self.X[0,feature])):
                index_left = indexs[self.X[indexs, feature] != split_value]
                index_right = indexs[self.X[indexs, feature] == split_value]
            else:
                index_left = indexs[self.X[indexs, feature] < split_value]
                index_right = indexs[self.X[indexs, feature] >= split_value]
                
            if (len(index_left)==0) or (len(index_right)==0):
                score = -np.inf
                
            elif "auc" in self.criterion:
                auc_y = get_auc_y(index_left, index_right)
                auc_b = get_auc_b(index_left, index_right)
                if "sub" in self.criterion:
                    score = (1-self.orthogonality)*auc_y - self.orthogonality*auc_b
                elif "div" in self.criterion:
                    score = auc_y / auc_b
                    
            elif self.criterion == "faht":
                
                n = len(indexs)
                pos_n = sum(self.y[indexs]==self.y_pos)
                neg_n = n - pos_n
                pos_prob = pos_n/n
                neg_prob = neg_n/n
                entropy_parent = entropy([pos_prob, neg_prob], base=2)

                n_left = len(index_left)
                pos_n_left = sum(self.y[index_left]==self.y_pos)
                neg_n_left = n_left - pos_n_left
                pos_prob_left = pos_n_left/n_left
                neg_prob_left = neg_n_left/n_left
                entropy_left = entropy([pos_prob_left, neg_prob_left], base=2)

                n_right = len(index_right)
                pos_n_right = sum(self.y[index_right]==self.y_pos)
                neg_n_right = n_right - pos_n_right
                pos_prob_right = pos_n_right/n_right
                neg_prob_right = neg_n_right/n_right
                entropy_right = entropy([pos_prob_right, neg_prob_right], base=2)

                ig = entropy_parent - (
                    (n_left/n) * entropy_left + (n_right/n) * entropy_right
                )
                
                dr = sum((self.b[indexs]==self.b_neg) & (self.y[indexs]==self.y_neg)) # deprived rejected
                dg = sum((self.b[indexs]==self.b_neg) & (self.y[indexs]==self.y_pos)) # deprived granted
                fr = sum((self.b[indexs]==self.b_pos) & (self.y[indexs]==self.y_neg)) # favoured rejected
                fg = sum((self.b[indexs]==self.b_pos) & (self.y[indexs]==self.y_pos)) # favoured granted
                disc = (fg/(fg+fr)) - (dg/(dg+dr))

                dr_left = sum((self.b[index_left]==self.b_neg) & (self.y[index_left]==self.y_neg)) # deprived rejected
                dg_left = sum((self.b[index_left]==self.b_neg) & (self.y[index_left]==self.y_pos)) # deprived granted
                fr_left = sum((self.b[index_left]==self.b_pos) & (self.y[index_left]==self.y_neg)) # favoured rejected
                fg_left = sum((self.b[index_left]==self.b_pos) & (self.y[index_left]==self.y_pos)) # favoured granted                
                if (fg_left+fr_left)==0:
                    disc_left = (dg_left/(dg_left+dr_left))
                elif (dg_left+dr_left)==0:
                    disc_left = (fg_left/(fg_left+fr_left)) 
                else:
                    disc_left = (fg_left/(fg_left+fr_left)) - (dg_left/(dg_left+dr_left))
                
                dr_right = sum((self.b[index_right]==self.b_neg) & (self.y[index_right]==self.y_neg)) # deprived rejected
                dg_right = sum((self.b[index_right]==self.b_neg) & (self.y[index_right]==self.y_pos)) # deprived granted
                fr_right = sum((self.b[index_right]==self.b_pos) & (self.y[index_right]==self.y_neg)) # favoured rejected
                fg_right = sum((self.b[index_right]==self.b_pos) & (self.y[index_right]==self.y_pos)) # favoured granted
                if (fg_right+fr_right)==0:
                    disc_right = (dg_right/(dg_right+dr_right))
                elif (dg_right+dr_right)==0:
                    disc_right = (fg_right/(fg_right+fr_right)) 
                else:
                    disc_right = (fg_right/(fg_right+fr_right)) - (dg_right/(dg_right+dr_right))
                
                fg = abs(disc) - ( (n_left/n) * abs(disc_left) + (n_right/n) * abs(disc_right))
                if (fg==0):
                    fg = 1 # FIG=IG*FG, and when FG=0, authors state FIG=IG --> FG=1 since FIG=IG*FG -> FIG=IG

                elif np.isnan(fg):
                    fg = -np.inf
                    ig = 1
                    
                score = ig * fg # fair information gain
                
            elif self.criterion in ["entropy", "ig"]:
                
                n = len(indexs)
                pos_n = sum(self.y[indexs]==self.y_pos)
                neg_n = n - pos_n
                pos_prob = pos_n/n
                neg_prob = neg_n/n
                entropy_parent = entropy([pos_prob, neg_prob], base=2)

                n_left = len(index_left)
                pos_n_left = sum(self.y[index_left]==self.y_pos)
                neg_n_left = n_left - pos_n_left
                pos_prob_left = pos_n_left/n_left
                neg_prob_left = neg_n_left/n_left
                entropy_left = entropy([pos_prob_left, neg_prob_left], base=2)

                n_right = len(index_right)
                pos_n_right = sum(self.y[index_right]==self.y_pos)
                neg_n_right = n_right - pos_n_right
                pos_prob_right = pos_n_right/n_right
                neg_prob_right = neg_n_right/n_right
                entropy_right = entropy([pos_prob_right, neg_prob_right], base=2)

                ig = entropy_parent - (
                    (n_left/n) * entropy_left + (n_right/n) * entropy_right
                )
                score = ig # information gain
            
            elif self.criterion == "fg":
                
                n = len(indexs)
                dr = sum((self.b[indexs]==self.b_neg) & (self.y[indexs]==self.y_neg)) # deprived rejected
                dg = sum((self.b[indexs]==self.b_neg) & (self.y[indexs]==self.y_pos)) # deprived granted
                fr = sum((self.b[indexs]==self.b_pos) & (self.y[indexs]==self.y_neg)) # favoured rejected
                fg = sum((self.b[indexs]==self.b_pos) & (self.y[indexs]==self.y_pos)) # favoured granted
                disc = (fg/(fg+fr)) - (dg/(dg+dr))
                
                n_left = len(index_left)
                dr_left = sum((self.b[index_left]==self.b_neg) & (self.y[index_left]==self.y_neg)) # deprived rejected
                dg_left = sum((self.b[index_left]==self.b_neg) & (self.y[index_left]==self.y_pos)) # deprived granted
                fr_left = sum((self.b[index_left]==self.b_pos) & (self.y[index_left]==self.y_neg)) # favoured rejected
                fg_left = sum((self.b[index_left]==self.b_pos) & (self.y[index_left]==self.y_pos)) # favoured granted                
                if (fg_left+fr_left)==0:
                    disc_left = (dg_left/(dg_left+dr_left))
                elif (dg_left+dr_left)==0:
                    disc_left = (fg_left/(fg_left+fr_left)) 
                else:
                    disc_left = (fg_left/(fg_left+fr_left)) - (dg_left/(dg_left+dr_left))
                
                n_right = len(index_right)
                dr_right = sum((self.b[index_right]==self.b_neg) & (self.y[index_right]==self.y_neg)) # deprived rejected
                dg_right = sum((self.b[index_right]==self.b_neg) & (self.y[index_right]==self.y_pos)) # deprived granted
                fr_right = sum((self.b[index_right]==self.b_pos) & (self.y[index_right]==self.y_neg)) # favoured rejected
                fg_right = sum((self.b[index_right]==self.b_pos) & (self.y[index_right]==self.y_pos)) # favoured granted
                if (fg_right+fr_right)==0:
                    disc_right = (dg_right/(dg_right+dr_right))
                elif (dg_right+dr_right)==0:
                    disc_right = (fg_right/(fg_right+fr_right)) 
                else:
                    disc_right = (fg_right/(fg_right+fr_right)) - (dg_right/(dg_right+dr_right))
                
                fg = abs(disc) - (
                    (n_left/n)*abs(disc_left) + (n_right/n)*abs(disc_right)
                )
                
                if np.isnan(fg):
                    fg = -np.inf
                score = fg # fairness gain
            
            elif "kamiran" in self.criterion:
                n = len(indexs)
                pos_n = sum(self.y[indexs]==self.y_pos)
                neg_n = n - pos_n
                pos_prob = pos_n/n
                neg_prob = neg_n/n
                entropy_parent = entropy([pos_prob, neg_prob], base=2)

                n_left = len(index_left)
                pos_n_left = sum(self.y[index_left]==self.y_pos)
                neg_n_left = n_left - pos_n_left
                pos_prob_left = pos_n_left/n_left
                neg_prob_left = neg_n_left/n_left
                entropy_left = entropy([pos_prob_left, neg_prob_left], base=2)

                n_right = len(index_right)
                pos_n_right = sum(self.y[index_right]==self.y_pos)
                neg_n_right = n_right - pos_n_right
                pos_prob_right = pos_n_right/n_right
                neg_prob_right = neg_n_right/n_right
                entropy_right = entropy([pos_prob_right, neg_prob_right], base=2)

                igc = entropy_parent - (
                    (n_left/n) * entropy_left + (n_right/n) * entropy_right
                )

                pos_n = sum(self.b[indexs]==self.b_pos)
                neg_n = n - pos_n
                pos_prob = pos_n/n
                neg_prob = neg_n/n
                entropy_parent = entropy([pos_prob, neg_prob], base=2)

                n_left = len(index_left)
                pos_n_left = sum(self.b[index_left]==self.b_pos)
                neg_n_left = n_left - pos_n_left
                pos_prob_left = pos_n_left/n_left
                neg_prob_left = neg_n_left/n_left
                entropy_left = entropy([pos_prob_left, neg_prob_left], base=2)

                n_right = len(index_right)
                pos_n_right = sum(self.b[index_right]==self.b_pos)
                neg_n_right = n_right - pos_n_right
                pos_prob_right = pos_n_right/n_right
                neg_prob_right = neg_n_right/n_right
                entropy_right = entropy([pos_prob_right, neg_prob_right], base=2)
                
                igs = entropy_parent - (
                    (n_left/n) * entropy_left + (n_right/n) * entropy_right
                )
                
                if "add" in self.criterion:
                    score = igc + igs
                
                if "sub" in self.criterion:
                    score = igc - igs
                    
                if "div" in self.criterion:
                    score = igc / igs
                    
            return score   
                
        # return best (sscore, feature, split_value) dependant on criterion and indexs
        def get_best_split(indexs):
            if self.criterion=="auc_sub":
                base_score = (1-self.orthogonality)*0.5 - self.orthogonality*0.5
            elif self.criterion=="auc_div":
                base_score = 1
            else:
                base_score = 0 # because other methods use "gain" (they already measure the difference)
            best_score = copy(base_score)
            candidate_splits = get_candidate_splits(indexs)
            for feature in candidate_splits:
                for split_value in candidate_splits[feature]:
                    score = evaluate_split(feature, split_value, indexs)
                    #print(score)
                    if score > best_score:
                        best_score = score
                        best_feature = feature
                        best_split_value = split_value
            if (best_score==base_score):
                best_score, best_feature, best_split_value = -np.inf, np.nan, np.nan
            return best_score, best_feature, best_split_value 
        
        # recursively grow the actual tree ---> {split1: {...}}
        def build_tree(indexs, step=0, old_score=-np.inf, new_score=-np.inf):     
            step = copy(step)
            indexs = copy(indexs)
            tree={}
            if (                
                len(np.unique(self.y[indexs]))==1 or ( # no need to split if there is alreadyd only 1 y class
                len(indexs)<=self.min_leaf) or ( # minimum number to consider a node as a leaf
                #new_score<old_score) or ( # if score is lower after split
                step>=self.max_depth) # if we've reached the max depth in the tree
            ):
                return indexs

            else:
                score, feature, split_value = get_best_split(indexs)
                old_score = copy(new_score)
                new_score = copy(score)
                
                if new_score==-np.inf: ## in case no more feature values exist for splitting
                    return indexs
                
                left_indexs = indexs[self.X[indexs, feature]<split_value]
                right_indexs = indexs[self.X[indexs, feature]>=split_value]
                
                if (len(left_indexs)==0) or (len(right_indexs)==0):
                    return indexs
                
                else:
                    step += 1
                    tree[(feature, split_value)] = {
                        "<": build_tree(left_indexs, step=copy(step), old_score=copy(old_score), new_score=copy(new_score)),
                        ">=":  build_tree(right_indexs, step=copy(step), old_score=copy(old_score), new_score=copy(new_score))
                    }

                    return tree
        
        self.tree = build_tree(self.samples)
        del self.X
        self.is_fit=True   
       
    def predict_proba(self, X):

        def get_probas_dict(tree, X, indexs=np.array([]), probas_dict={}):

            indexs = np.array(range(X.shape[0])) if len(indexs)==0 else indexs
            if type(tree)==type({}):
                feature, value = list(tree.keys())[0]
                left_indexs = indexs[X[indexs, feature]<value]
                sub_tree = tree[(feature, value)]["<"]
                probas_dict = get_probas_dict(sub_tree, X, left_indexs, probas_dict)
                right_indexs = indexs[X[indexs, feature]>=value]
                sub_tree = tree[(feature, value)][">="]
                probas_dict = get_probas_dict(sub_tree, X, right_indexs, probas_dict)
                return probas_dict

            else:
                index = copy(tree)
                sub_y = self.y[index]
                proba = sum(sub_y)/len(sub_y)
                if proba in probas_dict:
                    probas_dict[proba] += indexs.tolist()
                else:
                    probas_dict[proba] = indexs.tolist()
                return probas_dict

        proba = np.repeat(0.0, X.shape[0])
        probas_dict = get_probas_dict(self.tree, X)
        for proba_value in probas_dict:
            proba_index = np.array(probas_dict[proba_value])
            proba[proba_index] =  proba_value
        
        probas = proba.reshape(-1,1)
        probas = np.concatenate((1-probas, probas), axis=1)
        
        return probas
    
    def predict(self, X):
        
        def predict_proba(self, X):

            def get_probas_dict(tree, X, indexs=np.array([]), probas_dict={}):

                indexs = np.array(range(X.shape[0])) if len(indexs)==0 else indexs
                if type(tree)==type({}):
                    feature, value = list(tree.keys())[0]
                    left_indexs = indexs[X[indexs, feature]<value]
                    sub_tree = tree[(feature, value)]["<"]
                    probas_dict = get_probas_dict(sub_tree, X, left_indexs, probas_dict)
                    right_indexs = indexs[X[indexs, feature]>=value]
                    sub_tree = tree[(feature, value)][">="]
                    probas_dict = get_probas_dict(sub_tree, X, right_indexs, probas_dict)
                    return probas_dict

                else:
                    index = copy(tree)
                    sub_y = self.y[index]
                    proba = sum(sub_y)/len(sub_y)
                    if proba in probas_dict:
                        probas_dict[proba] += indexs.tolist()
                    else:
                        probas_dict[proba] = indexs.tolist()
                    return probas_dict

            proba = np.repeat(0.0, X.shape[0])
            probas_dict = get_probas_dict(self.tree, X)
            for proba_value in probas_dict:
                proba_index = np.array(probas_dict[proba_value])
                proba[proba_index] =  proba_value

            probas = proba.reshape(-1,1)
            probas = np.concatenate((1-probas, probas), axis=1)

            return probas
        
        probas = predict_proba(X)[:,1] 
        predicts = np.repeat(0, X.shape[0])
        predicts[probas>=self.pred_th] = 1
        
        return predicts
    
    def __str__(self):
        string = "FairDecisionTreeClassifier():" + "\n" + \
                "  is_fit=" + str(self.is_fit) + "\n" + \
                "  n_bins=" + str(self.n_bins) + "\n" + \
                "  min_leaf=" + str(self.min_leaf) + "\n" + \
                "  max_depth=" + str(self.max_depth) + "\n" + \
                "  n_samples=" + str(self.n_samples) + "\n" + \
                "  criterion=" + str(self.criterion) + "\n" + \
                "  max_features=" + str(self.max_features) + "\n" + \
                "  bias_method=" +str(self.bias_method) + "\n" + \
                "  orthogonality=" +str(self.orthogonality) + "\n" + \
                "  bootstrap=" +str(self.bootstrap) + "\n" + \
                "  random_state=" + str(self.random_state) + "\n" + \
                "  compound_bias_method=" + str(self.compound_bias_method)
        return string

    def __repr__(self):
        string = "FairDecisionTreeClassifier():" + "\n" + \
                "  is_fit=" + str(self.is_fit) + "\n" + \
                "  n_bins=" + str(self.n_bins) + "\n" + \
                "  min_leaf=" + str(self.min_leaf) + "\n" + \
                "  max_depth=" + str(self.max_depth) + "\n" + \
                "  n_samples=" + str(self.n_samples) + "\n" + \
                "  criterion=" + str(self.criterion) + "\n" + \
                "  max_features=" + str(self.max_features) + "\n" + \
                "  bias_method=" +str(self.bias_method) + "\n" + \
                "  orthogonality=" +str(self.orthogonality) + "\n" + \
                "  bootstrap=" +str(self.bootstrap) + "\n" + \
                "  random_state=" + str(self.random_state) + "\n" + \
                "  compound_bias_method=" + str(self.compound_bias_method)
        return string
  
  
class FairRandomForestClassifier():
    def __init__(self, n_estimators=100, n_jobs=100,
        n_bins=2, min_leaf=1, max_depth=2, n_samples=1.0, max_features="auto", 
        bootstrap=True, random_state=42, criterion="auc_sub", bias_method="avg", 
        compound_bias_method="avg", orthogonality=.5):
        """A fair random forest classifier.
        
        This random forest classifier enables to optimize both with regards to y 
        and b. In the case of orthogonality = 0, fully optimise for y (making it 
        an ordinary random forest classifier). In the case of orthogonality=1, 
        complete make the output of the learned classifier independent towards 
        the bias group.
        
        Parameters:
        - n_bins: Number of bins used to discretise continuous variables.
        - min_leaf: The minimum number of samples required to be in each leaf.
        - max_depth: The maximum depth of each tree.
        - n_samples: The number (int) or fraction (float) of samples to 
            bootstrap.
        - criterion: Either "entropy" or "auc_sub" score criterion for splitting.
        - max_features: Number (int) or proportion (float) of features to 
            bootstrap. Alternatively, use "auto" or "sqrt" to take the square 
            root of the total number of features to bootstrap, or "log" or 
            "log2" to use the log2 of the total number of features.
        - bias_method: OvR approach for multi-categorical bias attribute, can be 
            any of "avg", "w_avg", or "xtr".
        - orthogonality: Float between 0-1 (both endpoints inclusive), see 
            description.
        - bootstrap: Whether to bootstrap with replacement (True) or sample 
            without replacement (False).
        - random_state
        - compount_bias_method: Aggregation approach for multiple bias 
            attributes. Can be any of "avg", "xtr".
        """
        self.is_fit = False
        self.n_bins = n_bins
        self.n_jobs = n_jobs
        self.min_leaf = min_leaf
        self.max_depth = max_depth
        self.n_samples = n_samples
        self.criterion = criterion
        self.max_features = max_features
        self.bias_method = bias_method
        self.orthogonality = orthogonality
        self.bootstrap = bootstrap
        self.random_state = random_state        
        self.n_estimators = n_estimators
        self.compound_bias_method = compound_bias_method
        

        # Generating FairRandomForest
        dts = [
            FairDecisionTreeClassifier(
                n_bins=self.n_bins,
                min_leaf=self.min_leaf,
                max_depth=self.max_depth,
                n_samples=self.n_samples,
                criterion=self.criterion,
                random_state=self.random_state+i,
                max_features=self.max_features,
                bias_method=self.bias_method,
                orthogonality=self.orthogonality,
                bootstrap=self.bootstrap,
                compound_bias_method=self.compound_bias_method,
            )
            for i in range(self.n_estimators)
        ]
        self.trees = dts
        
    def fit(self, X, y, s, verbose=False):
      
        def batch(iterable, n_jobs=1):
            if n_jobs==-1:
                n_jobs = multiprocessing.cpu_count()
            l = len(iterable)
            n = int(np.ceil(l / n_jobs))
            for ndx in range(0, l, n):
                yield iterable[ndx:min(ndx + n, l)]

        def fit_trees_parallel(i, dt_batches, X, y, s):
            dt_batch = dt_batches[i]
            fit_dt_batch = []
            for dt in tqdm(dt_batch, desc=str(i), disable=not verbose):
                dt.fit(X, y, s)
                fit_dt_batch.append(dt)
            return fit_dt_batch
    
        dts = self.trees
        dt_batches = list(batch(dts, n_jobs=self.n_jobs))
        fit_dt_batches = Parallel(n_jobs=self.n_jobs)(
            delayed(fit_trees_parallel)(
                i, 
                dt_batches, 
                X, 
                y, 
                s, 
            ) for i in (range(len(copy(dt_batches))))
        )
        fit_dts = [tree 
                   for fit_dt_batch in fit_dt_batches 
                   for tree in fit_dt_batch]
        self.trees = fit_dts
        self.fit = True 
    
    def predict_proba(self, X):
        def predict_proba_parallel(dt_batch, X, i):
            probas = []
            for tree in dt_batch:
                probas.append(tree.predict_proba(X)[:,1])
            return np.array(probas)
        
        def batch(iterable, n_jobs=1):
            if n_jobs==-1:
                n_jobs = multiprocessing.cpu_count()
            l = len(iterable)
            n = ceil(l / n_jobs)
            for ndx in range(0, l, n):
                yield iterable[ndx:min(ndx + n, l)]
                
        if not self.fit:
            warnings.warn("Forest has not been fit(X,y,s)")
        
        else:
            dt_batches = list(batch(self.trees, n_jobs=self.n_jobs))
            
            y_preds = Parallel(n_jobs=self.n_jobs)(
                delayed(predict_proba_parallel)(
                    copy(dt_batches[i]),
                    copy(X),
                    copy(i),
                ) for i in range(len(dt_batches))
            )
            
            y_prob = y_preds[0]
            for i in range(1, len(y_preds)): 
                y_prob = np.concatenate((y_prob, y_preds[i]), axis=0)
            probas = np.mean(y_prob, axis=0).reshape(-1,1)
            probas = np.concatenate((1-probas, probas), axis=1)
            return probas
    
    def predict(self, X):
        def predict_parallel(tree, X):
            return tree.predict(X)
        if not self.fit:
            warnings.warn("Forest has not been fit(X,y,s)")

        else:
            # Predicting
            y_preds = Parallel(n_jobs=self.n_jobs)(
                delayed(predict_parallel)(
                    tree, X
                ) for tree in self.trees
            )
            y_preds = np.array(y_preds)
            # adding an "extra tree" with all positives so that ties are 
            # considered positive (just like >= th)
            predictions = mode(
                np.concatenate((
                    y_preds,
                    (
                        np.repeat(self.y_pos, y_preds.shape[1])
                        .reshape(1, y_preds.shape[1])
                    )
                ), axis=0))[0][0]
            return predictions


def get_filepath(theta, max_depth, n_bins, outer_fold, inner_fold=None):
    if inner_fold is not None:
        prefix = f'inner_{outer_fold}_{inner_fold}'
    else:
        prefix = f'outer_{outer_fold}'
    if n_bins == 2:
        base = f'theta_{theta:.1f}_max_depth_{max_depth}_n_bins_{int(n_bins):02}'
    else:
        base = f'theta_{theta}_max_depth_{max_depth}_n_bins_{int(n_bins):02}'
    return f'cache/FRF/{prefix}_{base}'


def classify(Xys, folds, theta, max_depth, n_bins, outer_fold, inner_fold=None):
    filepath = get_filepath(theta, max_depth, n_bins, outer_fold, inner_fold)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if os.path.isfile(filepath + '.json'):
        with open(filepath + '.json') as file:
            return json.load(file)
    else:
        print(f"Did not find {filepath + '.json'}")

    if inner_fold is not None:
        train_idx = folds[outer_fold][inner_fold]['train']
        val_idx = folds[outer_fold][inner_fold]['val']
        train = Xys.loc[train_idx]
        test = Xys.loc[val_idx]
    else:            
        test_idx = folds[outer_fold][0]['test']
        test = Xys.loc[test_idx]
        train = Xys[~Xys.index.isin(test_idx)]
    
    X_train = np.ascontiguousarray(train['X'].values)
    y_train = np.ascontiguousarray(train['y'].values.ravel())
    s_train = train['s'].values.ravel()
    X_test = np.ascontiguousarray(test['X'].values)
    y_test = np.ascontiguousarray(test['y'].values.ravel())
    s_test = test['s'].values.ravel()
    
    vt = sklearn.feature_selection.VarianceThreshold() 
    vt.fit(X_train)
    X_train = vt.transform(X_train)
    X_test = vt.transform(X_test) 
    
    if not os.path.isfile(filepath + '.pkl'):
        clf = FairRandomForestClassifier(orthogonality=theta, max_depth=max_depth)
        fit_start = time.time()
        clf.fit(X_train, y_train, s_train)
        fit_time = time.time() - fit_start
        clf.fit_time = fit_time 
        joblib.dump(clf, filepath + '.pkl')
    else:
        clf = joblib.load(filepath + '.pkl')
        fit_time = clf.fit_time
    probs = clf.predict_proba(X_test)[:,1]              
    auc_y = sklearn.metrics.roc_auc_score(y_test, probs)
    auc_s = sklearn.metrics.roc_auc_score(s_test, probs)
    auc_s = max([auc_s, 1-auc_s]) 
    
    result = {
        'auc_y': auc_y, 
        'auc_s': auc_s, 
        'fit_time': fit_time, 
        'clf': 'FRF',
        'theta': theta,
        'max_depth': max_depth,
        'n_bins': n_bins,
        'outer_fold': outer_fold
    }
    if inner_fold is not None:
        result['inner_fold'] = inner_fold
        
    with open(filepath + '.json', 'w') as file:
        json.dump(result, file)
    return result


def main(out_file: typing.Optional[str] = typer.Argument(None)):
    xys_file = 'data/Xys.pkl'
    folds_file = 'data/folds.json'
    
    assert os.path.isfile(xys_file)
    assert os.path.isfile(folds_file)
    if out_file:
        assert not os.path.isfile(out_file)
    
    Xys = pd.read_pickle(xys_file)
    with open(folds_file) as file:
        folds = json.load(file)
        
    random.seed(42)
    random.shuffle(iterations)
    data = [
        classify(Xys, folds, theta, max_depth, n_bins, outer_fold, inner_fold)
        for theta, max_depth, n_bins, outer_fold, inner_fold 
        in tqdm(iterations, miniters=1, mininterval=0)
    ]

    inner_cv_results = (
        pd.DataFrame(data)
        .assign(div=lambda x: x['auc_y'] / x['auc_s'],
                min=lambda x: x['auc_y'] - x['auc_s'])
        .groupby(['outer_fold', 'max_depth', 'theta', 'n_bins'])
        .agg({'auc_y': 'mean', 'auc_s': 'mean', 'div': 'mean', 'min': 'mean'})
        .reset_index()
    )

    best_hp = {
        objective: (
            inner_cv_results
            .sort_values(objective, ascending=objective == 'auc_s')
            .groupby(['outer_fold'])
            .first()
            .reset_index()
            [['outer_fold', 'max_depth', 'theta', 'n_bins']]
        )
        for objective in ('auc_y', 'auc_s', 'div', 'min')
    }

    unique_hp = (
        pd.concat(best_hp.values())
        .drop_duplicates()
        .to_dict(orient='records')
    )

    for p in tqdm(unique_hp, mininterval=0, miniters=1):
        classify(Xys, folds, p['theta'], p['max_depth'], p['n_bins'], 
                 p['outer_fold'])

    results = []
    for objective, parameters in best_hp.items():
        data = list()
        for p in parameters.to_dict(orient='records'):
            filepath = get_filepath(
                p['theta'], p['max_depth'], p['n_bins'], p['outer_fold']
            )
            with open(filepath + '.json') as file:
                data.append(json.load(file))
        results.append(pd.DataFrame(data).assign(objective=objective))
    
    results = (
        pd.concat(results, ignore_index=True)
        .groupby('objective')
        .agg({'auc_y': 'mean', 'auc_s': 'mean'})
        .reset_index()
    )    
    
    if out_file is None:
        return results
    else:
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        results.to_json(out_file, orient='records', indent=4)    
  
if __name__ == '__main__':
    typer.run(main)