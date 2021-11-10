import datetime
import json
import operator
import os
import typing

import joblib
import networkx as nx
import numpy as np
import pandas as pd
import sklearn.feature_selection
import sklearn.metrics
import sklearn.model_selection
from tqdm.auto import tqdm

from .logger import logger
from .helper_functions import largest_weakly_connected_component

portcalls_filepath = 'data/portcalls_v3.pkl'
inspections_filepath = 'data/inspections_v3.pkl'
X_filepath = 'data/X_v3.pkl'
y_filepath = 'data/y_v3.pkl'
s_filepath = 'data/s_v3.pkl'

start_date = datetime.datetime(2014, 1, 1)
end_date = datetime.datetime(2017, 12, 1)

# def get_filename(
#         clf, version, max_depth, n_bins, model, portcalls_added, flags_added=None
#     ):
#     assert 0 < version <= 3
#     assert model in ('A', 'B')
#     assert portcalls_added in (True, False)
#     assert flags_added in (None, True, False)
    
#     clf = clf.upper()
    
#     if flags_added is None:
#         flags = 'o'
#     elif flags_added:
#         flags = '+'
#     else:
#         flags = '-'
#     if portcalls_added:
#         portcalls = '+'
#     else:
#         portcalls = '-'
    
#     filepath = f"cache/{clf}_v{version}/"
#     filename = f"{max_depth:02}_{n_bins:02}_{model}_{portcalls}_{flags}.pkl"
#     return filepath + filename

def get_portcalls(
        cache_file: str = 'data/portcalls_v3.pkl',
        portcalls_file: str = 'data/raw/portcalls_new.csv', 
        performance_file: str = 'data/raw/parismou_performancelist.json',
        start_date: typing.Optional[datetime.datetime] = start_date,
        end_date: typing.Optional[datetime.datetime] = end_date,
        minimum_calls_per_ship: int = 2,
    ) -> pd.DataFrame:
    """
    Import the portcalls dataset. If present at cache_file, use that file.
    
    Arguments:
    - portcalls_file: filepath to portcalls file
    - performance_file: filepath to performance file
    - start_date: Consider only portcalls after this date.
    - end_date: Consider only portcalls before this date.
    - verbose
    
    Returns:
    - DataFrame containing the following columns: 
        - arrival: datetime
        - departure: datetime
        - port
        - ship: str containing imo_number
        - risk: 0 (low) / 1 (medium) / 2 (high)
        - flag
    """
    if os.path.isfile(cache_file):
        portcalls = joblib.load(cache_file)
    else: 
        assert os.path.isfile(portcalls_file)
        assert os.path.isfile(performance_file)

        # Read in files
        with open(performance_file) as file:
            performance_list = json.load(file)

        portcalls = (
            pd.read_csv(portcalls_file, sep=';', low_memory=False, index_col=0)
            .rename(
                columns={
                    'ATA_LT': 'arrival',
                    'ATD_LT': 'departure',
                    'Port.Location.Name': 'port',
                    'IMO.Number': 'ship',
                    'X.ATA..Ship.Risk.Profile': 'risk',
                    'X.ATA..Ship.Flag.Code': 'flag',
                }
            )
            .assign(
                arrival=lambda x: pd.to_datetime(x['arrival']),
                departure=lambda x: pd.to_datetime(x['departure']),
                flag=lambda x: x['flag'].replace(performance_list) != False,
            )
            .replace({'risk': {'HRS': 2, 'SRS': 1, 'LRS': 0}})
            .astype({'port': str, 'ship': str, 'risk': 'Int64'})
            .fillna({'risk': 1})
            .sort_values('arrival')
            [['arrival', 'departure', 'port', 'ship', 'risk', 'flag']]
        )
        portcalls.to_pickle(cache_file)
    
    if start_date is not None:
        portcalls = portcalls.loc[lambda x: (start_date <= x['arrival'])]
    if end_date is not None:
        portcalls = portcalls.loc[lambda x: (x['arrival'] <= end_date)]
    
    # Select only portcalls from ships that have at least two portcalls.
    ships, counts = np.unique(portcalls['ship'], return_counts=True)
    ships_with_multiple_portcalls = set(
        ships[counts >= minimum_calls_per_ship]
    )
    
    portcalls = (
        portcalls.loc[lambda x: x['ship'].isin(ships_with_multiple_portcalls)]
    )
    return portcalls
    
def get_inspections(
        cache_file: str = 'data/inspections_v3.pkl',
        inspections_file: str = 'data/raw/inspections.csv',
        start_date: typing.Optional[datetime.datetime] = start_date,
        end_date: typing.Optional[datetime.datetime] = end_date
    ) -> pd.DataFrame:
    if os.path.isfile(cache_file):
        inspections = pd.read_pickle(cache_file)  
        assert isinstance(inspections, pd.DataFrame)
        return inspections
    
    assert os.path.isfile(inspections_file)
    
    inspections = pd.read_csv(
        inspections_file,
        engine='python',
        index_col=0,
        skipfooter=1,
        parse_dates=['DateOfFirstVisit', 'ShipKeelLayingDate'],
        dtype={'WasDetained': bool}
    )
    inspections.sort_values('DateOfFirstVisit', inplace=True)
    
    if start_date is not None:
        inspections = (
            inspections.loc[lambda x: start_date <= x['DateOfFirstVisit']]
        )
    if end_date is not None:
        inspections = (
            inspections.loc[lambda x: x['DateOfFirstVisit'] <= end_date]
        )
    
    inspections.to_pickle(cache_file)
    return inspections

def get_ship_labels(portcalls, inspections):
    """
    Get all ships with their labels, where 0 marks a ship with no deficiencies,
    1 a ship with only deficiencies (no detentions) and 1 a ship which had at
    some point a detention. Only the highest number is stored.
    
    Arguments:
        - portcalls_df
        - inspections_df
        
    Returns:
        - ships_with_label: DataFrame with columns ship (IMO number) and label.
    """
    ships_with_portcalls = set(portcalls['ship'].unique())
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
    
def select(ships_with_label):
    """
    Select 10% at random from the ships. This is done in a stratified way such 
    that the column label is stratified.
    
    Arguments:
        - ships_with_label: DataFrame containing columns ship and label.
    
    Returns:
        - selected: np.array containing the ships that were selected.
        - not_selected: np.array containing the ships that were not selected.
    """
    # Select random 10%.
    not_selected, selected = sklearn.model_selection.train_test_split(
        ships_with_label['ship'], test_size=.1, random_state=42, 
        stratify=ships_with_label['label']
    )
    return selected.values, not_selected.values

def get_network(portcalls: pd.DataFrame) -> nx.DiGraph:
    """
    Obtain network with all used node attributes stored in graph:
    - degree
    - in-degree
    - out-degree
    - strength
    - in-strength
    - out-strength
    - closeness centrality (weighted and unweighted)
    - betweenness centrality (weighted and unweighted)
    - eigenvector centrality (weighted and unweighted)

    These node measures can be obtained as follows:
    > pd.DataFrame.from_dict(
        dict(network_base.nodes(data=True)), orient='index')
    """
    # Get edgelist
    edgelist = pd.concat(
        [
            pd.DataFrame(
                {
                'source': portcalls['port'].shift(1),
                'target': portcalls['port'],
                'duration': (
                    portcalls['arrival'] - portcalls['departure'].shift(1)
                ),
                'weight': len(portcalls) - 1,
                'distance': 1/(len(portcalls) - 1),
                }
            ).dropna()
            for _, ship_df in portcalls.groupby('ship')
        ]
    )

    # Get graph
    G = nx.from_pandas_edgelist(edgelist, edge_attr=True, 
                                create_using=nx.DiGraph)
    # Select largest weakly connected component
    return largest_weakly_connected_component(G)

def get_travel_times(df):
    return (
        df
        .groupby('ship')
        .apply(lambda x: (x.arrival - x.departure.shift(1)).dt.seconds // 60)
        .dropna()
        .values
    )

def get_port_stays(df):
    return ((df['departure'] - df['arrival']).dt.seconds // 60).values

def hist(x, bins):
    """Return the values of the histogram."""
    return np.histogram(x, bins)[0]

def set_node_attributes(G: nx.DiGraph) -> nx.DiGraph:
    # Set node attributes
    nx.set_node_attributes(G, dict(G.degree), 'degree')
    nx.set_node_attributes(G, dict(G.in_degree), 'in_degree')
    nx.set_node_attributes(G, dict(G.out_degree), 'out_degree')
    nx.set_node_attributes(G, dict(G.degree(weight='weight')), 'strength') #type: ignore
    nx.set_node_attributes(G, dict(G.in_degree(weight='weight')), 
                           'in_strength'), 
    nx.set_node_attributes(G, dict(G.out_degree(weight='weight')), 
                           'out_strength') #type: ignore
    nx.set_node_attributes(G, 
                           nx.closeness_centrality(G, wf_improved=False), 
                           'closeness')
    nx.set_node_attributes(G, 
                           nx.betweenness_centrality(G, normalized=False), 
                           'betweenness')
    nx.set_node_attributes(G, 
                           nx.eigenvector_centrality(G, max_iter=100_000), 
                           'eigenvector')
    nx.set_node_attributes(G, 
                           nx.closeness_centrality(G, distance='distance', 
                                                   wf_improved=False), 
                           'closenss_weighted')
    nx.set_node_attributes(G, 
                           nx.betweenness_centrality(G, weight='weight', 
                                                     normalized=False), 
                           'betweenness_weighted')
    nx.set_node_attributes(G, 
                           nx.eigenvector_centrality(G, weight='weight', 
                                                     max_iter=100_000), 
                           'eigenvectors_weighted')
    return G

def feature_engineering(portcalls_selected, 
                        portcalls_not_selected, 
                        number_of_bins=10) -> pd.DataFrame:    
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
    
    network_filepath = 'data/network.pkl'
    if os.path.isfile(network_filepath):
        logger.debug(f'Load network from {network_filepath}')
        G = joblib.load(network_filepath)
    else:
        logger.debug(f'Get network.')
        G = get_network(portcalls_selected)
        logger.debug(f'Store network')
        G.to_pickle(network_filepath)
    
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
    logger.debug('Get node measures.')
    node_measures = pd.DataFrame.from_dict(dict(G.nodes(data=True)), 
                                           orient='index')
    
    # Get edge_measures (bins), by taking the four default operations on the values 
    # of the two nodes.
    operations = {
        'sub': operator.sub, 
        'add': operator.add, 
        'mul': operator.mul, 
        'div': div
    }    

    logger.debug('Get all values for binning')
    all_base_values = {'travel_time': get_travel_times(portcalls_selected), 
                       'port_stay': get_port_stays(portcalls_selected)}
    
    for operation_str, operation_func in tqdm(operations.items()):
        node_pair_measures = pd.DataFrame.from_dict(
            {(u,v): operation_func(node_measures.loc[u], node_measures.loc[v]) 
            for u, v in G.edges()}, #type: ignore
            orient='index'
        )
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
    # Prepopulate X
    X_dict = {'port_stay': {}, 
             'travel_time': {}, 
             'both_nodes_missing': {}, 
             'one_node_missing': {},
             'number_of_journeys': {}}

    other_ships = portcalls_not_selected['ship'].unique()
    for measure in node_measures:
        for operation in operations:
            X_dict[(measure,operation)] = {
                ship: np.tile(0, number_of_bins) for ship in other_ships
            } 

    # Get dict X
    for ship, ship_df in tqdm(portcalls_not_selected.groupby('ship'), 
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
            if (port_u not in G) and (port_v not in G):
                X_dict['both_nodes_missing'][ship] += 1 / (n-1)
            elif (port_u not in G) ^ (port_v not in G):
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

    X = pd.concat(
        {
            measure: pd.DataFrame.from_dict(measure_dict, orient='index') 
            for measure, measure_dict in X_dict.items()
        }, 
        axis='columns'
    )
    return X

def get_Xys(
        X_filepath: str = 'data/X_v3.pkl',
        y_filepath: str = 'data/y_v3.pkl',
        s_filepath: str = 'data/s_v3.pkl',
        selected_filepath: str = 'data/selected.npy',
        not_selected_filepath: str = 'data/not_selected.npy',
        start_date: datetime.datetime = start_date,
        end_date: datetime.datetime = end_date,
        numpy: bool = True
    ):
    
    cache_filepaths = (X_filepath, y_filepath, s_filepath)
    if all([os.path.isfile(file) for file in cache_filepaths]):
        X = joblib.load(X_filepath)
        y = joblib.load(y_filepath)
        s = joblib.load(s_filepath)
        assert all(X.index == s.index)
        assert all(y.index == s.index)
    else:
        # PORTCALLS
        portcalls = get_portcalls(start_date=start_date, end_date=end_date)
        inspections = get_inspections(start_date=start_date, end_date=end_date)

        # FEATURE ENGINEERING
        logger.debug("Get ship labels")
        ships_with_label = get_ship_labels(portcalls, inspections)

        logger.debug("Select ship labels")
        selected, not_selected = select(ships_with_label)
        np.save(selected_filepath, selected)
        np.save(not_selected_filepath, not_selected)

        logger.debug("Test selection")
        portcalls_selected = portcalls.loc[lambda x: x['ship'].isin(selected)]
        assert (portcalls_selected.groupby('ship').size() > 1).all()
        portcalls_not_selected = (
            portcalls.loc[lambda x: x['ship'].isin(not_selected)] #type: ignore
        )
        assert (portcalls_not_selected.groupby('ship').size() > 1).all()
        len_portcalls = len(portcalls_selected) + len(portcalls_not_selected)
        assert len(portcalls) == len_portcalls

        # X
        if not os.path.isfile(X_filepath):
            logger.debug("Start feature engineering.")
            X = feature_engineering(portcalls_selected, portcalls_not_selected)
            logger.debug(f'Store features at {X_filepath}')
            X.to_pickle(X_filepath)
        else:
            logger.debug(f'Load features at {X_filepath}')
            X = joblib.load(X_filepath)

        # y
        if not os.path.isfile(y_filepath):
            logger.debug("Start collecting labels.")
            inspected_ships_with_result = (
                inspections
                .groupby('IMO')
                ['WasDetained']
                .any()
                .replace({False: 1, True: 2})
            )
            y = pd.Series(
                {
                    ship: inspected_ships_with_result.get(ship, default=0) 
                    for ship in X.index
                }
            )
            logger.debug(f'Store labels at {y_filepath}')
            y.to_pickle(y_filepath)
        else:
            logger.debug(f"Load labels at {y_filepath}")
            y = joblib.load(y_filepath)
            
        # s
        if not os.path.isfile(s_filepath):
            logger.debug(f"Start collecting sensitive attributes")
            s = portcalls.groupby('ship')['flag'].last()[X.index].astype(int).astype(bool)
            logger.debug(f"Store sensitive attributes at {s_filepath}")
            s.to_pickle(s_filepath)
        else:
            logger.debug(f"Load sensitive attributes from {s_filepath}")
            s = joblib.load(s_filepath)
        
        # Some tests
        assert (X.index == y.index).all()
        assert (X.index == s.index).all()
    
    if numpy:
        X = np.ascontiguousarray(X.values)
        y = np.ascontiguousarray(y.values.ravel())
        s = np.ascontiguousarray(s.values.ravel())
    return X, y, s

def get_expert_labels() -> pd.Series:
    portcalls = get_portcalls()
    expert_labels = portcalls.groupby('ship')['risk'].last()
    return expert_labels

