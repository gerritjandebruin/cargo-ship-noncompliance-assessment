import numpy as np
import pandas as pd


def get_targets(inspections: pd.DataFrame,
                ships_classification: np.ndarray) -> pd.DataFrame:
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
    print(len(ships_classification))
    ships_with_label = list()
    for ship in ships_classification:
        if ship in ships_detained:
            ships_with_label.append(2)
        elif ship in ships_inspected:
            ships_with_label.append(1)
        else: # Ship not on deficiency list --> OK!
            ships_with_label.append(0)
    return pd.DataFrame(ships_with_label, index=ships_classification)