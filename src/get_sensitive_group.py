import numpy as np
import pandas as pd


def get_sensitive_group(portcalls: pd.DataFrame, 
                        ships: np.ndarray) -> pd.Series:
    """Get for every ships whether it belongs to the sensitive group. Only the
    last known flag is used.
    """
    return (
        portcalls
        .groupby('ship')
        ['flag']
        .last()
        [ships]
        .astype(int)
        .astype(bool)
    )