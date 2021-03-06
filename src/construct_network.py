import networkx as nx
import pandas as pd


def construct_network(portcalls: pd.DataFrame) -> nx.DiGraph:
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
    assert 'port' in portcalls.columns
    assert 'arrival' in portcalls.columns
    assert 'departure' in portcalls.columns
    assert 'ship' in portcalls.columns
    assert all(
        (portcalls['departure']-portcalls['arrival']).dropna() > pd.Timedelta(0)
    )
    
    objs = list()
    for _, ship_df in portcalls.groupby('ship'):
        duration = ship_df['arrival'] - ship_df['departure'].shift(1)
        assert all(duration.dropna() >= pd.Timedelta(0)), (
            print(ship_df['arrival'], ship_df['departure'].shift(1)), duration)
        obj = pd.DataFrame(
            {
                'source': ship_df['port'].shift(1),
                'target': ship_df['port'],
                'duration': duration,
                'weight': len(ship_df) - 1,
                'distance': 1/(len(ship_df) - 1),
            }
        ).dropna()
        objs.append(obj)
    edgelist = pd.concat(objs)

    assert all(edgelist['duration'] >= pd.Timedelta(0))

    # Get graph
    G = nx.from_pandas_edgelist(edgelist, edge_attr=True, 
                                create_using=nx.DiGraph)    
    nx.set_node_attributes(G, dict(G.degree), 'degree')
    nx.set_node_attributes(G, dict(G.in_degree), 'in_degree')
    nx.set_node_attributes(G, dict(G.out_degree), 'out_degree')
    nx.set_node_attributes(G, dict(G.degree(weight='weight')), 'strength')
    nx.set_node_attributes(G, dict(G.in_degree(weight='weight')), 
                           'in_strength'), 
    nx.set_node_attributes(G, dict(G.out_degree(weight='weight')), 
                           'out_strength')
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