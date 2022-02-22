import os

import pandas as pd

def import_inspections(filepath: str) -> pd.DataFrame:
    """Load inspections from filepath, parse dates and sort on date of first 
    visit.
    """
    assert os.path.isfile(filepath)
    return (
        pd.read_csv(
            filepath,
            engine='python',
            index_col=0,
            skipfooter=1,
            parse_dates=['DateOfFirstVisit', 'ShipKeelLayingDate'],
            infer_datetime_format=True,
            dayfirst=True,
            dtype={'WasDetained': bool}
        )
        .sort_values('DateOfFirstVisit')
    )