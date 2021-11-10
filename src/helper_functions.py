import networkx as nx

def largest_weakly_connected_component(G: nx.DiGraph) -> nx.DiGraph:
    """Return largest weakly connected component."""
    gc = G.subgraph(max(nx.weakly_connected_components(G), key=len))
    return gc.copy()