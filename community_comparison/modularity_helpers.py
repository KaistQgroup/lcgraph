from typing import Dict, List, Set, Any, Optional, Iterable
import networkx as nx
from networkx.algorithms.community.quality import modularity as nx_modularity

def _make_partition_total(
    G: nx.Graph,
    raw_groups: Iterable[Iterable[Any]],
    deduplicate: bool = True,
    fill_missing_as_singletons: bool = True,
) -> List[Set[Any]]:
    """
    Make sure the partition:
      - has no empty groups,
      - has no duplicates,
      - covers *all* nodes of G (optionally fill missing as singletons),
      - is disjoint (drops overlaps conservatively by first-come).
    """
    all_nodes = set(G.nodes())
    parts: List[Set[Any]] = []
    seen_groups: Set[frozenset] = set()
    covered: Set[Any] = set()

    for grp in raw_groups:
        s = set(grp) & all_nodes
        if not s:
            continue
        if deduplicate:
            f = frozenset(s)
            if f in seen_groups:
                continue
            seen_groups.add(f)
        # enforce disjointness: remove already-covered nodes
        s -= covered
        if not s:
            continue
        parts.append(s)
        covered |= s

    if fill_missing_as_singletons:
        missing = all_nodes - covered
        for u in missing:
            parts.append({u})

    return parts

def partition_from_level_obj(net, level: int, G: nx.Graph) -> List[Set[Any]]:
    """
    Builds a *total* partition for `level` from your net.levels structure.
    Falls back to singletons for any nodes not listed at that level.
    """
    raw: List[Iterable[Any]] = []
    if hasattr(net, "levels") and (level in net.levels):
        # Your structure seems to store a list of "community sets" at each level,
        # each with a dict of communities whose `.nodes` are original node IDs.
        for cs in net.levels[level]:
            for _, comm in cs.get_communities_dict().items():
                raw.append(comm.nodes)  # Iterable of original nodes
    # Build a total, disjoint, full-cover partition
    return _make_partition_total(G, raw, deduplicate=True, fill_missing_as_singletons=True)

def modularity_for_partition(
    G: nx.Graph,
    communities: List[Set[Any]],
    weight: Optional[str] = "weight",
    resolution: float = 1.0,
) -> float:
    # Ensure undirected simple graph for NetworkX modularity (symmetrize if needed)
    if G.is_directed():
        H = nx.Graph(G)  # symmetrize
    else:
        H = G
    return nx_modularity(H, communities, weight=weight, resolution=resolution)
