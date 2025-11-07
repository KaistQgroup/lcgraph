from typing import Dict, List, Set, Any, Optional, Tuple
import networkx as nx
from networkx.algorithms.community import louvain_communities

def _as_simple_undirected(G: nx.Graph, weight_attr: str = "weight") -> nx.Graph:
    if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
        H = nx.Graph()
        for u, v, data in G.edges(data=True):
            w = float(data.get(weight_attr, 1.0))
            if H.has_edge(u, v):
                H[u][v][weight_attr] += w
            else:
                H.add_edge(u, v, **{weight_attr: w})
        H.add_nodes_from(G.nodes())
        return H
    if G.is_directed():
        G = nx.Graph(G)  # symmetrize
    return G

def _ensure_nonneg_weights(G: nx.Graph, weight_attr: str = "weight") -> None:
    for u, v, data in G.edges(data=True):
        w = float(data.get(weight_attr, 1.0))
        if w <= 0:
            data[weight_attr] = 1.0

def _contract_graph(
    G: nx.Graph,
    node2c: Dict[Any, int],
    weight_attr: str = "weight",
) -> Tuple[nx.Graph, Dict[int, Set[Any]]]:
    H = nx.Graph()
    supernodes: Dict[int, Set[Any]] = {}
    for u in G.nodes():
        c = node2c[u]
        supernodes.setdefault(c, set()).add(u)
        if c not in H:
            H.add_node(c)

    for u, v, data in G.edges(data=True):
        cu, cv = node2c[u], node2c[v]
        w = float(data.get(weight_attr, 1.0))
        if H.has_edge(cu, cv):
            H[cu][cv][weight_attr] += w
        else:
            H.add_edge(cu, cv, **{weight_attr: w})

    # accumulate self-loops
    for c, members in supernodes.items():
        w_in = 0.0
        for u in members:
            for v, d in G[u].items():
                if v in members:
                    w_in += float(d.get(weight_attr, 1.0))
        if w_in > 0.0:
            if H.has_edge(c, c):
                H[c][c][weight_attr] += w_in
            else:
                H.add_edge(c, c, **{weight_attr: w_in})

    return H, supernodes

def louvain_multi_level(
    G: nx.Graph,
    max_depth: Optional[int] = None,
    seed: Optional[int] = 123,
    resolution: float = 1.0,
    weight_attr: str = "weight",
) -> Dict[int, List[Set[Any]]]:
    G_curr = _as_simple_undirected(G, weight_attr=weight_attr)
    _ensure_nonneg_weights(G_curr, weight_attr=weight_attr)

    partitions_fine_to_coarse: List[List[Set[Any]]] = []
    maps_to_orig: List[Dict[Any, Set[Any]]] = []

    map_curr = {u: {u} for u in G_curr.nodes()}
    depth = 0

    while True:
        if G_curr.number_of_nodes() == 0:
            break

        comms = louvain_communities(
            G_curr, weight=weight_attr, resolution=resolution, seed=seed
        )
        # always record the current level
        partitions_fine_to_coarse.append(comms)
        maps_to_orig.append(map_curr)

        n_nodes = G_curr.number_of_nodes()
        n_comms = len(comms)

        # stop if no further coarsening is possible or max depth reached
        if (max_depth is not None and depth + 1 >= max_depth) or n_comms == n_nodes:
            break

        node2c = {u: i for i, cset in enumerate(comms) for u in cset}
        G_next, supernodes = _contract_graph(G_curr, node2c, weight_attr=weight_attr)
        map_next = {c: set().union(*(map_curr[u] for u in members))
                    for c, members in supernodes.items()}

        G_curr = G_next
        map_curr = map_next
        depth += 1

    # rebuild levels over original nodes; 0 = coarsest
    levels: Dict[int, List[Set[Any]]] = {}
    L = len(partitions_fine_to_coarse)
    if L == 0:
        return {0: [set(G.nodes())]}  # defensive fallback

    for li in range(L):
        idx_coarsest = L - 1 - li
        comms = partitions_fine_to_coarse[idx_coarsest]
        mp = maps_to_orig[idx_coarsest]
        level_sets: List[Set[Any]] = []
        for cset in comms:
            orig = set()
            for u in cset:
                orig |= mp[u]
            if orig:
                level_sets.append(orig)
        levels[li] = level_sets if level_sets else [set(G.nodes())]

    return levels
