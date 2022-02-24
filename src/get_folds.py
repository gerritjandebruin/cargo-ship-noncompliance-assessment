import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def get_folds(X: pd.DataFrame, y: pd.Series, s: pd.Series, 
              random_state: int=42) -> tuple[list, list]:
    """Provide the outer and inner folds for the given instances."""
    X = np.ascontiguousarray(X.values)
    y = np.ascontiguousarray(y.values.ravel())
    s = np.ascontiguousarray(s.values.ravel())
    
    ys = y + 3*s
    
    skf = StratifiedKFold(shuffle=True, random_state=random_state)
    outer_folds = list(enumerate(skf.split(X=X, y=ys)))
    inner_folds = [
        (
            (outer_idx, inner_idx), 
            (trainval_idx[train_idx], trainval_idx[val_idx])
        )
        for outer_idx, (trainval_idx, _) in outer_folds
        for inner_idx, (train_idx, val_idx)
        in enumerate(skf.split(X=X[trainval_idx], y=ys[trainval_idx]))
    ]
    return outer_folds, inner_folds