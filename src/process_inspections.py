import datetime

import numpy as np
import pandas as pd

from .constants import START_DATE, END_DATE


def process_inspections(inspections: pd.DataFrame, 
                        start_date: datetime.datetime = START_DATE, 
                        end_date: datetime.datetime = END_DATE):
    """Select only inspections occurring between start_date and end_date 
    (both inclusive).
    """
    cols = ['DateOfFirstVisit', 'IMO', 'WasDetained']
    assert all([col in inspections.columns for col in cols])
    assert inspections['WasDetained'].dtype == bool
    assert np.issubdtype(inspections['DateOfFirstVisit'], np.datetime64)
    
    return (
        inspections
        .loc[lambda x: start_date <= x['DateOfFirstVisit']]
        .loc[lambda x: x['DateOfFirstVisit'] <= end_date]
        .loc[:, ['IMO', 'WasDetained', 'DateOfFirstVisit']]
    )     