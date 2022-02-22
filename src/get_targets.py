import pandas as pd

def get_targets(portcalls: pd.DataFrame, 
                inspections: pd.DataFrame) -> pd.DataFrame:
    """
    Get all ships with their labels, where 0 marks a ship with no deficiencies,
    and 1 a ship with deficiencies..
        
    Returns:
        - ships_with_label: DataFrame with columns ship (IMO number) and label.
    """
    ships_inspected = set(inspections['IMO'].unique())
    ships_detained = set(
        inspections.loc[lambda x: x['WasDetained'], 'IMO'].unique()
    )
    
    ships_with_label = list()
    for ship in portcalls.ship.unique():
        if ship in ships_detained:
            ships_with_label.append({'ship': ship, 'label': 2})
        elif ship in ships_inspected:
            ships_with_label.append({'ship': ship, 'label': 1})
        else: # Ship not on deficiency list --> OK!
            ships_with_label.append({'ship': ship, 'label': 0})
    return pd.DataFrame(ships_with_label)