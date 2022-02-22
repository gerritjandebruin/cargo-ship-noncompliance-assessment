import pandas as pd


def import_portcalls(filepath: str, flag_performance: dict):
    """Load portcalls from filepath and do some cleaning steps:
    - Columns are renamed.
    - Date columns are parsed as such.
    - flag_code holds the flag of a ship
    - flag indicates the type of flag, white (0), grey (1), or black (2)
    - risk indicates whether the ship is high (2), medium (1) or low (0) risk
    - Portcalls are sorted on arrival date.
    """
    assert all([len(key) == 2 for key in flag_performance.keys()])
    assert all([0 <= value <= 2 and type(value) is int 
                for value in flag_performance.values()])
    
    return (
        pd.read_csv(
            filepath, sep=';', low_memory=False, index_col=0, 
            parse_dates=['ATA_LT', 'ATD_LT', 'X.ATA..Ship.Keel.Laying.Date'], 
            infer_datetime_format=True, dayfirst=True)
        .rename(
            columns={
                'ATA_LT': 'arrival',
                'ATD_LT': 'departure',
                'Port.Location.Name': 'port',
                'IMO.Number': 'ship',
                'X.ATA..Ship.Risk.Profile': 'risk',
                'X.ATA..Ship.Flag.Code': 'flag',
                'X.ATA..Ship.Keel.Laying.Date': 'keel_laying',
                'X.ATA..Ship.Type.Description': 'type',
                'X.ATA..Ship.Type.Is.High.Risk': 'type_high_risk',
                'X.ATA..Ship.Flag.Is.PMOU': 'flag_in_pmou',
                'X.ATA..Ship.Priority': 'priority'
                })
        .assign(
            flag_code=lambda x: x['flag'],
            flag=lambda x: x['flag'].replace(flag_performance))
        .dropna(subset=['flag'])
        .replace({'risk': {'HRS': 2, 'SRS': 1, 'LRS': 0}})
        .astype({'port': str, 'ship': str, 'risk': 'Int8', 'flag': 'int8'})
        .sort_values('arrival')
        [[
            'arrival', 'departure', 'port', 'ship', 'risk', 'flag', 
            'keel_laying', 'type', 'type_high_risk', 'flag_in_pmou', 
            'flag_code', 'Sent.At', 'priority'
        ]]
    )
