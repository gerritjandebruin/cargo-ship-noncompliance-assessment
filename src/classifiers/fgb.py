import multiprocessing
from copy import deepcopy as copy

import numpy as np
from joblib import Parallel, delayed
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm as tqdm_n


class FGBClassifier():
    
    ####################
    # TODO:
    # bootstrap
    # max_depth
    # max_features
    # inv weight OvR
    # multiple sens-attr
    ####################
    
    def __init__(self, n_estimators=100, learning_rate=1e0, theta=0.5, n_jobs=-1, verbose=True):
        self.theta = theta
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        
    def fit(self, X, y, s):

        def get_batches(iterable, n_jobs=-1):
            if n_jobs==-1:
                n_jobs = multiprocessing.cpu_count() # - 1 # -1 so that our laptop doesn't freeze
            l = len(iterable)
            n = int(np.ceil(l / n_jobs))
            for ndx in range(0, l, n):
                yield iterable[ndx:min(ndx + n, l)]

        def find_best_split_parallel(batch, X, y, s, z, idx, theta, learning_rate):
            y_auc = roc_auc_score(y, z)
            
            if len(s.shape)>1:
                ovr_s_auc = []
                for j in range(s.shape[1]):
                    s_auc = roc_auc_score(s[:, j], z)
                    s_auc = max(1-s_auc, s_auc)
                    ovr_s_auc.append(s_auc)
                s_auc = max(ovr_s_auc)
            else:
                s_auc = roc_auc_score(s, z)
                s_auc = max(1-s_auc, s_auc)
                
            base_score = (1-theta)*y_auc - theta*s_auc
            best_score = copy(base_score)    
            
            for split in batch:
                variable, value = split
                left_idx = idx[X[idx, variable]<value]
                right_idx = idx[X[idx, variable]>=value]
                left_n, right_n = len(left_idx), len(right_idx)
                if (left_n>0) and (right_n>0):
                    left_y, right_y = y[left_idx], y[right_idx]
                    left_s, right_s = s[left_idx], s[right_idx]
                    left_z, right_z = z[left_idx], z[right_idx] 
                    left_p, right_p = expit(left_z), expit(right_z)
                    left_z_increase = np.mean(left_y - left_p)*learning_rate
                    right_z_increase = np.mean(right_y - right_p)*learning_rate        
                    left_new_z = left_z + left_z_increase
                    right_new_z = right_z + right_z_increase
                    
                    y_auc = roc_auc_score(
                        left_y.tolist()+right_y.tolist(),
                        left_new_z.tolist()+right_new_z.tolist()
                    )
                    
                    if len(s.shape)>1:
                        ovr_s_auc = []
                        for j in range(s.shape[1]):
                            s_auc = roc_auc_score(
                                s[left_idx, j].tolist()+s[right_idx, j].tolist(),
                                left_new_z.tolist()+right_new_z.tolist()
                            )
                            s_auc = max(1-s_auc, s_auc)
                            ovr_s_auc.append(s_auc)
                        s_auc = max(ovr_s_auc)
                    else:
                        s_auc = roc_auc_score(
                            s[left_idx].tolist()+s[right_idx].tolist(),
                            left_new_z.tolist()+right_new_z.tolist()
                        )
                        s_auc = max(1-s_auc, s_auc)
                    
                    score = (1-theta)*y_auc - theta*s_auc
                    if score > best_score:
                        best_split = split
                        best_score = score
                        best_left_n = left_n
                        best_right_n = right_n
                        best_left_idx = left_idx
                        best_right_idx = right_idx
                        best_left_z_increase = left_z_increase
                        best_right_z_increase = right_z_increase
                        
            if best_score==base_score:
                best_split = np.nan
                best_score = -np.inf
                best_left_idx = np.nan
                best_right_idx = np.nan
                best_left_z_increase = np.nan
                best_right_z_increase = np.nan
            return best_left_z_increase, best_left_idx, best_right_z_increase, best_right_idx, best_split, best_score
        
        theta = self.theta
        n_jobs = self.n_jobs
        verbose = self.verbose
        n_estimators = self.n_estimators
        learning_rate = self.learning_rate

        n, m = X.shape
        z = np.repeat(0.0, n)
        idx = np.array(range(n))

        splits = [
            (variable, np.unique(X[idx, variable])[i])
            for variable in range(m)
                for i in range(len(np.unique(X[idx, variable])))
        ]
        batches = list(get_batches(splits, n_jobs=n_jobs))

        trees = []
        best_score=0
        while best_score!=-np.inf:
            if verbose:
                for i in tqdm_n(range(n_estimators), leave=False):
                    results = Parallel(n_jobs=n_jobs)(
                        delayed(find_best_split_parallel)(
                            batch, X, y, s, z, idx, theta, learning_rate
                        ) for batch in batches
                    )
                    best_left_z_increase, best_left_idx, best_right_z_increase, best_right_idx, best_split, best_score = sorted(
                        results, key=lambda x: x[-1]
                    )[-1]
                    if best_score!=-np.inf:
                        tree = {
                            "split": best_split,
                            0: best_left_z_increase,
                            1: best_right_z_increase,
                        }
                        trees.append(tree)
                        z[best_left_idx] = z[best_left_idx] + best_left_z_increase
                        z[best_right_idx] = z[best_right_idx] + best_right_z_increase

                        y_auc = roc_auc_score(
                            y[best_left_idx].tolist()+y[best_right_idx].tolist(),
                            z[best_left_idx].tolist()+z[best_right_idx].tolist(),
                        )
                        
                        if len(s.shape)>1:
                            ovr_s_auc = []
                            for j in range(s.shape[1]):
                                s_auc = roc_auc_score(
                                    s[best_left_idx, j].tolist()+s[best_right_idx, j].tolist(),
                                    z[best_left_idx].tolist()+z[best_right_idx].tolist(),
                                )
                                s_auc = max(1-s_auc, s_auc)
                                ovr_s_auc.append(s_auc)
                            s_auc = max(ovr_s_auc)
                        else:
                            s_auc = roc_auc_score(
                                s[best_left_idx].tolist()+s[best_right_idx].tolist(),
                                z[best_left_idx].tolist()+z[best_right_idx].tolist(),
                            )
                            s_auc = max(1-s_auc, s_auc)
                        
                        y_auc = round(y_auc, 4)
                        s_auc = round(s_auc, 4)
                        print_line = "y_AUC=" + str(y_auc) + "\ts_AUC=" + str(s_auc)
                        if len(s.shape)>1:
                            for j in range(s.shape[1]):
                                print_line += "\ts"+str(j+1)+"_AUC=" + str(round(ovr_s_auc[j], 4))
                        # sys.stdout.write("\r" + str(print_line)+"\t")
                        # sys.stdout.flush()
                        tqdm_n.write(print_line)
                    else:
                        break
            else:
                for i in range(n_estimators):
                    results = Parallel(n_jobs=n_jobs)(
                        delayed(find_best_split_parallel)(
                            batch, X, y, s, z, idx, theta, learning_rate
                        ) for batch in batches
                    )
                    best_left_z_increase, best_left_idx, best_right_z_increase, best_right_idx, best_split, best_score = sorted(
                        results, key=lambda x: x[-1]
                    )[-1]
                    if best_score!=-np.inf:
                        tree = {
                            "split": best_split,
                            0: best_left_z_increase,
                            1: best_right_z_increase,
                        }
                        trees.append(tree)
                        z[best_left_idx] = z[best_left_idx] + best_left_z_increase
                        z[best_right_idx] = z[best_right_idx] + best_right_z_increase
                    else:
                        break
            best_score=-np.inf
        self.trees = trees
    
    def predict_proba(self, X):
        z = np.repeat(0.0, X.shape[0])
        idx = np.array(range(X.shape[0]))
        for tree in self.trees:
            feature, value = tree["split"]
            left_z_increase = tree[0]
            right_z_increase = tree[1]

            left_idx = idx[X[idx, feature]<value]
            right_idx = idx[X[idx, feature]>=value]

            z[left_idx] = z[left_idx] + left_z_increase
            z[right_idx] = z[right_idx] + right_z_increase
        
        
        p = expit(z)
        proba = p.reshape(-1,1)
        return np.concatenate((1-proba, proba), axis=1)
    