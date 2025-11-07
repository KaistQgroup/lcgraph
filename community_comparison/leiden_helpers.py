from typing import List, Set, Hashable, Optional
import networkx as nx
import igraph as ig
import leidenalg as la
import inspect

def _nx_to_igraph(G: nx.Graph, nodes_subset: Optional[Set[Hashable]] = None) -> tuple[ig.Graph, dict, dict]:
    """
    Convert a (sub)graph to igraph.
    Returns: (ig_graph, nx_id_to_idx, idx_to_nx_id)
    """
    if nodes_subset is None:
        nodes = list(G.nodes())
    else:
        nodes = list(nodes_subset)

    nx_id_to_idx = {n: i for i, n in enumerate(nodes)}
    idx_to_nx_id = {i: n for n, i in nx_id_to_idx.items()}

    # print(nx_id_to_idx)
    # print(idx_to_nx_id)

    # Build edge list restricted to nodes
    edges = []
    weights = []
    for u, v, data in G.edges(nodes, data=True):
        if u in nx_id_to_idx and v in nx_id_to_idx:
            edges.append((nx_id_to_idx[u], nx_id_to_idx[v]))
            # Optional: weight support
            w = data.get("weight", 1.0)
            weights.append(float(w))

    g = ig.Graph(n=len(nodes), edges=edges, directed=G.is_directed())
    if weights:
        g.es["weight"] = weights

    return g, nx_id_to_idx, idx_to_nx_id


def _partition_kwargs_for_cpm(resolution: float) -> dict:
    """Handle leidenalg versions that use gamma vs resolution_parameter."""
    # Prefer introspection (works for most versions)
    sig = inspect.signature(la.CPMVertexPartition.__init__)
    if "resolution_parameter" in sig.parameters:
        return {"resolution_parameter": resolution}
    if "gamma" in sig.parameters:
        return {"gamma": resolution}
    # Fallback: just try resolution_parameter; user can adjust if needed
    return {"resolution_parameter": resolution}


def leiden_communities_nx(
    G: nx.Graph,
    nodes_subset: Optional[Set[Hashable]] = None,
    objective: str = "CPM",   # or "CPM"
    resolution: float = 1.0,         # gamma for CPM; ignored for modularity
    weights_attr: Optional[str] = "weight",
    seed: int = 123,
    n_iterations: int = -1,
) -> List[Set[Hashable]]:
    """
    Run Leiden and return communities as a list of sets of original NetworkX node IDs.
    Mirrors the shape of networkx.algorithms.community.louvain_communities.
    """
    g, nx_id_to_idx, idx_to_nx_id = _nx_to_igraph(G, nodes_subset)

    # Pick the quality function
    if objective.lower() == "modularity":
        partition_type = la.ModularityVertexPartition
        partition_kwargs = {}
    elif objective.upper() == "CPM":
        partition_type = la.CPMVertexPartition
        partition_kwargs = _partition_kwargs_for_cpm(resolution)
    else:
        raise ValueError("objective must be 'modularity' or 'CPM'")

    # Attach weights if present
    weights = g.es["weight"] if (weights_attr and "weight" in g.es.attribute_names()) else None

    # Run Leiden
    part = la.find_partition(
        g,
        partition_type,
        weights=weights,
        seed=seed,
        n_iterations=n_iterations,
        **partition_kwargs,
    )

    # Convert back to original node IDs
    comms: List[Set[Hashable]] = []
    for community in part:
        comms.append({idx_to_nx_id[idx] for idx in community})

    return comms


