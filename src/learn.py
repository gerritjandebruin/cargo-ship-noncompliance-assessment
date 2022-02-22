from os.path import isfile
from time import time

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

from src.fair_random_forest import FairRandomForestClassifier


def learn(X: pd.DataFrame, y: pd.DataFrame, s: pd.DataFrame, outer_folds: list, 
          inner_folds: list) -> pd.DataFrame:
    """Apply the entire machine learning procedure.
    
    Arguments: 
    - X: A m*n dataframe containing features, that is used as input for 
        classifier
    - y: A boolean vector of length n, containing the targets
    - s: A boolean vector of length n, indicating whether a sample belongs to 
        sensitive group.
    - outer_folds, inner_folds: Result of src.get_folds.
        
    Returns a pd.DataFrame containing the performance over all folds.
    """
    assert all(X.index == y.index)
    assert all(X.index == s.index)
    
    # Convert X, y, s to np.arrays for compatibility reasons.
    X = np.ascontiguousarray(X.values)
    y = np.ascontiguousarray(y.values.ravel())
    s = np.ascontiguousarray(s.values.ravel())
    
    params = [
        (int(max_depth), int(n_bins), float(orthogonality))
        for n_bins in (2,)
        for max_depth in np.arange(1, 11) 
        for orthogonality in np.linspace(0, 1, 11)
    ]
    
    # Learn on every outer fold
    iterations = [
        (max_depth, n_bins, ortho, fold, trainval_idx, test_idx)
        for max_depth, n_bins, ortho in params
        for fold, (trainval_idx, test_idx) in outer_folds
        if not isfile(f'models/outer_folds/{max_depth}-{ortho:.2f}-{n_bins}-{fold}.pkl')
    ]
    
    for max_depth, n_bins, ortho, fold, trainval_idx, test_idx in tqdm(iterations):
        X_trainval = X[trainval_idx]
        y_trainval = y[trainval_idx]
        s_trainval = s[trainval_idx]
        
        vt = VarianceThreshold()
        vt.fit(X_trainval)
        X_trainval = vt.transform(X_trainval)
        
        clf = FairRandomForestClassifier(
            orthogonality=ortho, max_depth=max_depth, n_bins=n_bins)
        start_fit = time()
        clf.fit(X_trainval, y_trainval, s_trainval)
        clf.fit_time = time() - start_fit
        fp = f'models/outer_folds/{max_depth}-{ortho:.2f}-{n_bins}-{fold}.pkl'
        joblib.dump(clf, fp)
        
    # Learn on every inner fold
    iterations = [
        (max_depth, n_bins, ortho, outer_fold, inner_fold, train_idx, val_idx)
        for max_depth, n_bins, ortho in params
        for (outer_fold, inner_fold), (train_idx, val_idx) in inner_folds
        if not isfile(f'models/inner_folds/{max_depth}-{ortho:.2f}-{n_bins}-{outer_fold}-{inner_fold}.pkl')
    ]
    for max_depth, n_bins, ortho, outer_fold, inner_fold, train_idx, val_idx in tqdm(iterations):    
        X_train = X[train_idx]
        y_train = y[train_idx]
        s_train = s[train_idx]
        vt = VarianceThreshold()
        vt.fit(X_train)
        X_train = vt.transform(X_train)
        clf = FairRandomForestClassifier(
            orthogonality=ortho, max_depth=max_depth, n_bins=n_bins)
        start_fit = time()
        clf.fit(X_train, y_train, s_train)
        clf.fit_time = time() - start_fit
        fp = f'models/inner_folds/{max_depth}-{ortho:.2f}-{n_bins}-{outer_fold}-{inner_fold}.pkl'
        joblib.dump(clf, fp)
        
    # Predict on all outer folds
    iterations = [
        (max_depth, n_bins, ortho, fold, trainval_idx, test_idx)
        for max_depth, n_bins, ortho in params
        for fold, (trainval_idx, test_idx) in outer_folds
        if not isfile(f'models/outer_folds/{max_depth}-{ortho:.2f}-{n_bins}-{fold}.npy')
    ]

    for max_depth, n_bins, ortho, fold, trainval_idx, test_idx in tqdm(iterations):        
        X_trainval = X[trainval_idx]
        X_test = X[test_idx]
        
        vt = VarianceThreshold()
        vt.fit(X_trainval)
        X_trainval = vt.transform(X_trainval)
        X_test = vt.transform(X_test)
        
        fp = f'models/outer_folds/{max_depth}-{ortho:.2f}-{n_bins}-{fold}'
        clf = joblib.load(f'{fp}.pkl')
        y_score = clf.predict_proba(X_test)[:,1]
        np.save(f'{fp}.npy', y_score)
    
    # Predict on all inner folds
    iterations = [
        (max_depth, n_bins, ortho, outer_fold, inner_fold, 
        train_idx, val_idx)
        for max_depth, n_bins, ortho in params
        for (outer_fold, inner_fold), (train_idx, val_idx) in inner_folds
        if not isfile(f'models/inner_folds/{max_depth}-{ortho:.2f}-{n_bins}-{outer_fold}-{inner_fold}.npy')
    ]
    for max_depth, n_bins, ortho, outer_fold, inner_fold, train_idx, val_idx in tqdm(iterations):
        
        X_train = X[train_idx]
        X_val = X[val_idx]
        
        vt = VarianceThreshold()
        vt.fit(X_train)
        X_train = vt.transform(X_train)
        X_val = vt.transform(X_val)
        
        fp = f'models/inner_folds/{max_depth}-{ortho:.2f}-{n_bins}-{outer_fold}-{inner_fold}'
        clf = joblib.load(f'{fp}.pkl')
        y_score = clf.predict_proba(X_val)[:,1]
        np.save(f'{fp}.npy', y_score)
    
    # Measure performance for every outer loop
    iterations = [
        (max_depth, n_bins, orthogonality, outer_fold, inner_fold, 
        train_idx, val_idx)
        for max_depth, n_bins, orthogonality in params
        for (outer_fold, inner_fold), (train_idx, val_idx) in inner_folds
    ]

    performance_all_candidates = list()
    for max_depth, n_bins, ortho, outer_fold, inner_fold, train_idx, val_idx in tqdm(iterations):
        fp = f'models/inner_folds/{max_depth}-{ortho:.2f}-{n_bins}-{outer_fold}-{inner_fold}'
        y_score = np.load(f'{fp}.npy')
        
        y_val = y[val_idx]
        s_val = s[val_idx]
        auc_y = roc_auc_score(y_val, y_score)
        auc_s = roc_auc_score(s_val, y_score)
        auc_s = max(auc_s, 1-auc_s)
        
        performance_this_run = dict(
            max_depth=max_depth, n_bins=n_bins, orthogonality=ortho,
            outer_fold=outer_fold, inner_fold=inner_fold, auc_y=auc_y, 
            auc_s=auc_s)
    performance_all_candidates.append(performance_this_run)
    return pd.DataFrame(performance_all_candidates)