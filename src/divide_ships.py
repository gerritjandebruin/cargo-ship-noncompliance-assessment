import pandas as pd
from sklearn.model_selection import train_test_split


def divide_ships(portcalls: pd.DataFrame, inspections: pd.DataFrame):
    """Get all ships from the portcalls and divide them either into the set used
    for classification or an hold-out set. The sampling procedure is stratified
    with respect to the inspection outcome.
    
    Returns:
        - ships_classification (90% random sample)
        - ships_network (10% random sample)
    """
    assert all([col in portcalls.columns for col in ('ship', )])
    assert all([col in inspections.columns for col in ('WasDetained', 'IMO')])
    
    ships_inspected = set(portcalls['ship'].unique())
    ships_detained = set(
        inspections
        .loc[lambda x: x['WasDetained'], 'IMO']
        .unique())
    ships = portcalls['ship'].unique()
    stratify = list()
    
    for ship in portcalls['ship'].unique():
        if ship in ships_detained:
            stratify.append(2)
        elif ship in ships_inspected:
            stratify.append(1)
        else: # Ship not on deficiency list --> OK!
            stratify.append(0)
    
    return train_test_split(ships, test_size=.1, random_state=42, 
                            stratify=stratify) 