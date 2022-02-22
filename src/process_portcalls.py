import datetime

import numpy as np
import pandas as pd

from .constants import START_DATE, END_DATE

def process_portcalls(portcalls: pd.DataFrame, 
                      start_date: datetime.datetime = START_DATE,
                      end_date: datetime.datetime = END_DATE):
    """Process the portcalls further:
    - Flag column is made binary; False indicates a white flag and True a 
        non-white flag.
    - Only portcalls with an arrival date between start_date and end_date are
        returned.
    - Only portcalls from ships that occur at least twice (between start_date 
        and end_date) are considered. 
    
    """
    cols = ['risk', 'flag', 'arrival', 'departure', 'ship', 'port']
    assert all([col in portcalls.columns for col in cols])
    assert portcalls['risk'].dtype == 'Int8'
    assert portcalls['flag'].dtype == 'int8'
    assert np.issubdtype(portcalls['arrival'], np.datetime64)
    assert np.issubdtype(portcalls['departure'], np.datetime64)
    assert all(portcalls['departure'] > portcalls['arrival'])
    
    return (
        portcalls
        .fillna({'risk': 1}) # See notebook
        .astype({'flag': 'bool'})
        .loc[lambda x: start_date <= x['arrival']]
        .loc[lambda x: x['arrival'] <= end_date]
        .loc[lambda x: x['ship'].duplicated(keep=False)]
        [['arrival', 'departure', 'port', 'ship', 'risk', 'flag']]
    )
    