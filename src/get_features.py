import operator
import os

import networkx as nx
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .constants import NUMBER_OF_BINS
from .largest_connected_component import largest_weakly_connected_component
from .logger import logger


def get_travel_times(portcalls: pd.DataFrame) -> np.ndarray:
    """Return an array containing all travel times in seconds for all ships."""
    cols = ('ship', 'arrival', 'departure')
    assert all([col in portcalls.columns for col in cols])
    return (
        portcalls
        .groupby('ship')
        .apply(lambda x: (x.arrival - x.departure.shift(1)).dt.seconds // 60)
        .dropna()
        .values
    )
    
def get_port_stays(portcalls: pd.DataFrame) -> np.ndarray:
    """Return an array containing for each portcall how long the ship was in the 
    port."""
    df = ((portcalls['departure'] - portcalls['arrival']).dt.seconds // 60)
    return df.values

def hist(x: np.ndarray, bins: int) -> np.ndarray:
    """Return the values of the histogram."""
    return np.histogram(x, bins)[0]

def get_features(
        portcalls_network: pd.DataFrame, 
        portcalls_classification: pd.DataFrame, 
        network: nx.DiGraph,
        number_of_bins: int = NUMBER_OF_BINS) -> pd.DataFrame:
    """Do the entire feature engineering. These features serve as input for the 
    fair random forest classifier. It models behaviour of the ships, which is
    derived from the portcall data.
    """   
    def div(x, y):
        """Same as x / y, except that it handles the division by zero."""
        large_int = 2000 # Largest encountered value is 1184
        if isinstance(x, pd.Series):
            return (x / y).replace({np.inf: large_int, np.nan: 0})
        else:
            if x == 0:
                return 0
            elif y != 0:
                return x / y
            else:
                return large_int
    
    # Get the following node measures:
    # - degree
    # - in-degree
    # - out-degree
    # - strength
    # - in-strength
    # - out-strength
    # - closeness centrality (weighted and unweighted)
    # - betweenness centrality (weighted and unweighted)
    # - eigenvector centrality (weighted and unweighted)
    network = largest_weakly_connected_component(network)
    node_dict = dict(network.nodes(data=True))
    node_measures = pd.DataFrame.from_dict(node_dict, orient='index')
    
    # Get edge_measures (bins), by taking the four default operations on the 
    # values of the two nodes.
    operations = {
        'sub': operator.sub, 
        'add': operator.add, 
        'mul': operator.mul, 
        'div': div
    }    

    # Get all values for making histogram
    all_base_values = {
        'travel_time': get_travel_times(portcalls_network), 
        'port_stay': get_port_stays(portcalls_network)
    }
    
    for operation_str, operation_func in tqdm(operations.items()):
        node_pair_measures_dict = {
            (u,v): operation_func(node_measures.loc[u], node_measures.loc[v]) 
            for u, v in network.edges()
        }
        node_pair_measures = pd.DataFrame.from_dict(node_pair_measures_dict,
                                                    orient='index')
        for measure_str, measure_series in tqdm(node_pair_measures.iteritems()):
            all_base_values[(measure_str, operation_str)] = (
                measure_series.values
            )
    # all_base_values has now the following keys:
    # (edge features)
    # - travel_time
    # - port_stay
    # (node pair features)
    # - degree (times 4 operations)
    # - in-degree (times 4 operations)
    # - out-degree (times 4 operations)
    # - strength (times 4 operations)
    # - in-strength (times 4 operations)
    # - out-strength (times 4 operations)
    # - closeness centrality (weighted and unweighted) (times 4 operations)
    # - betweenness centrality (weighted and unweighted) (times 4 operations)
    # - eigenvector centrality (weighted and unweighted) (times 4 operations)
    
    # Get binning
    logger.debug('Get binning')
    bins = {}
    for measurement_str, measurement_values in tqdm(all_base_values.items()):
        bins[measurement_str] = np.histogram_bin_edges(measurement_values, 
                                                       number_of_bins)
    # Prepopulate X (X will in the end provide all features as input for ML 
    # model)
    X_dict = {
        'port_stay': {}, 
        'travel_time': {}, 
        'both_nodes_missing': {}, 
        'one_node_missing': {},
        'number_of_journeys': {}
    }

    ships_classification = portcalls_classification['ship'].unique()
    for measure in node_measures:
        for operation in operations:
            X_dict[(measure,operation)] = {
                ship: np.tile(0, number_of_bins) 
                for ship in ships_classification
            } 

    # Get dict X
    for ship, ship_df in tqdm(portcalls_classification.groupby('ship'), 
                              leave=False):
        n = len(ship_df)
        X_dict['number_of_journeys'][ship] = n
        assert n > 1
        X_dict['port_stay'][ship] = (
            hist(get_port_stays(ship_df), bins['port_stay']) / n
        )
        X_dict['travel_time'][ship] = (
            hist(get_travel_times(ship_df), bins['travel_time']) / (n-1)
        )
        X_dict['both_nodes_missing'][ship] = 0
        X_dict['one_node_missing'][ship] = 0
        journeys = (
            pd.DataFrame({'u': ship_df['port'], 'v': ship_df['port'].shift(1)})
            .dropna()
        )
        for _, port_u, port_v in journeys.itertuples():
            if (port_u not in network) and (port_v not in network):
                X_dict['both_nodes_missing'][ship] += 1 / (n-1)
            elif (port_u not in network) ^ (port_v not in network):
                X_dict['one_node_missing'][ship] += 1 / (n-1)
            else:
                # All node pair measures
                for operation_str, operation_func in operations.items():
                    measures = operation_func(node_measures.loc[port_u], 
                                              node_measures.loc[port_v])
                    for measure, value in measures.iteritems():
                        feature_name = (measure,operation_str)
                        old_value = X_dict[feature_name][ship]
                        feature_value = (
                            (old_value + hist(value, bins[feature_name])) / 
                            (n-1)
                        )
                        X_dict[feature_name][ship] = feature_value
    # Now, we have the following features:
    # (edge features)
    # - travel_time
    # - port_stay
    # (node pair features)
    # - two nodes missing
    # - one node missing
    # - degree (times 4 operations)
    # - in-degree (times 4 operations)
    # - out-degree (times 4 operations)
    # - strength (times 4 operations)
    # - in-strength (times 4 operations)
    # - out-strength (times 4 operations)
    # - closeness centrality (weighted and unweighted) (times 4 operations)
    # - betweenness centrality (weighted and unweighted) (times 4 operations)
    # - eigenvector centrality (weighted and unweighted) (times 4 operations)
    X = {
        measure: pd.DataFrame.from_dict(measure_dict, orient='index') 
        for measure, measure_dict in X_dict.items()
    }
    return pd.concat(X, axis='columns')