import os

import pandas as pd


def import_un_locode(filepaths: list[str]):
    """Import UN Locode files."""
    names = [
        'Ch', 'LOCODE_country', 'LOCODE_city', 'Name', 'NameWoDiacritics', 
        'SubDiv', 'Function', 'Status', 'Date', 'IATA', 'Coordinates', 
        'Remarks']
    dfs = list()
    for filepath in filepaths:
        assert os.path.isfile(filepath)
        dfs.append(pd.read_csv(filepath, encoding_errors='ignore', names=names))
        
    df = pd.concat(dfs).dropna(subset=['Coordinates'])
    
    Latitude = list()
    Longitude = list()
    for lat_str, long_str in df['Coordinates'].str.split():
        sign_lat = -1 if lat_str[-1] == 'S' else 1
        sign_long = -1 if long_str[-1] == 'W' else 1
        Latitude.append(sign_lat * float(lat_str[:-1]) / 100)
        Longitude.append(sign_long * float(long_str[:-1]) / 100)
    labels = df['LOCODE_country'] + df['LOCODE_city']
    
    return (
        pd.DataFrame({'x': Longitude, 'y': Latitude, 'label': labels})
        .drop_duplicates(subset='label')
        .set_index('label')
    )
    