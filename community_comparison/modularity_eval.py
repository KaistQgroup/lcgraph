from __future__ import annotations
import os, csv, json, networkx as nx
from typing import Dict, List, Set, Any, Optional, Tuple, Callable
from networkx.algorithms.community.quality import modularity as nx_modularity


def _make_partition_total(
    G: nx.Graph,
    raw_groups,
    *,
    deduplicate: bool = True,
    fill_missing_mode: str = "remainder",  # "remainder" | "singletons" | "drop"
) -> List[Set[Any]]:
    """
    Turn possibly overlapping/duplicated groups into a full disjoint partition.
    - Ensures every node is assigned at most once.
    - Covers all nodes, using the chosen fill strategy for leftovers.
    """
    all_nodes = set(G.nodes())
    cleaned = [set(grp) & all_nodes for grp in raw_groups]
    cleaned = [c for c in cleaned if c]
    cleaned.sort(key=len, reverse=True)  # big groups first

    parts: List[Set[Any]] = []
    seen: set = set()
    covered: Set[Any] = set()

    for s in cleaned:
        f = frozenset(s)
        if deduplicate and f in seen:
            continue
        s = s - covered
        if not s:
            continue
        parts.append(s)
        covered |= s
        if deduplicate:
            seen.add(f)

    remainder = all_nodes - covered
    if remainder:
        if fill_missing_mode == "remainder":
            parts.append(remainder)
        elif fill_missing_mode == "singletons":
            for u in remainder:
                parts.append({u})
        elif fill_missing_mode == "drop":
            pass
        else:
            raise ValueError("fill_missing_mode must be 'remainder', 'singletons', or 'drop'")

    # final sanity: if we didn't drop, we should cover all nodes
    if fill_missing_mode != "drop":
        assert set().union(*parts) == all_nodes, "Partition does not cover all nodes."
    return parts


def partition_from_level_obj(
    net,
    level: int,
    G: nx.Graph,
    *,
    fill_missing_mode: str = "remainder",
) -> List[Set[Any]]:
    """
    Extract communities from net.levels[level] (list of CommunitySet),
    concatenate into a disjoint global partition with full coverage.
    Expects CommunitySet.get_communities_dict() with objects that have .nodes.
    """
    raw = []
    if hasattr(net, "levels") and (level in net.levels):
        for cs in net.levels[level]:
            for _, comm in cs.get_communities_dict().items():
                raw.append(comm.nodes)
    return _make_partition_total(G, raw, deduplicate=True, fill_missing_mode=fill_missing_mode)


def modularity_for_partition(
    G: nx.Graph,
    communities: List[Set[Any]],
    *,
    weight: Optional[str] = "auto",
    resolution: float = 1.0,
) -> float:
    """
    Compute modularity (undirected). If G is directed, symmetrize first.
    weight="auto" → use 'weight' if any edge has it; else None.
    """
    # robust weight detection
    if weight == "auto":
        weight_attr = None
        for _, _, d in G.edges(data=True):
            if "weight" in d:
                weight_attr = "weight"
                break
    else:
        weight_attr = weight

    H = nx.Graph(G) if G.is_directed() else G
    return nx_modularity(H, communities, weight=weight_attr, resolution=resolution)


def _top5_sizes(parts: List[Set[Any]]) -> List[int]:
    return sorted((len(p) for p in parts), reverse=True)[:5]


def modularity_all_levels(
    G: nx.Graph,
    net,
    *,
    weight: Optional[str] = "auto",
    resolution: float = 1.0,
    max_level: Optional[int] = None,
    fill_missing_mode: str = "remainder",
) -> Dict[int, Dict[str, Any]]:
    """
    Return:
      { level: {'num_communities': int, 'modularity': float, 'top5': List[int]} }
    """
    present_levels = sorted(getattr(net, "levels", {}).keys())
    if max_level is not None:
        present_levels = [L for L in present_levels if L <= max_level]

    out: Dict[int, Dict[str, Any]] = {}
    for L in present_levels:
        parts = partition_from_level_obj(net, L, G, fill_missing_mode=fill_missing_mode)
        q = float("nan")
        if len(parts) >= 2:
            q = modularity_for_partition(G, parts, weight=weight, resolution=resolution)
        out[L] = {
            "num_communities": len(parts),
            "modularity": q,
            "top5": _top5_sizes(parts),
        }
    return out

# ========= ORCHESTRATION / CSV =========

def run_modularity_methods_and_collect(
    G: nx.Graph,
    methods: List[Tuple[str, Callable[[nx.Graph], Any]]],
    *,
    weight: Optional[str] = "auto",
    resolution: float = 1.0,
    max_level: Optional[int] = None,
    fill_missing_mode: str = "remainder",
) -> List[Dict[str, Any]]:
    """
    methods: list of (name, factory) where factory(G) -> model instance with .levels
    Returns rows for CSV: method, level, num_communities, modularity, top_5_community
    """
    rows: List[Dict[str, Any]] = []
    for name, factory in methods:
        model = factory(G)                 # your constructors already build levels
        stats = modularity_all_levels(
            G, model,
            weight=weight,
            resolution=resolution,
            max_level=max_level,
            fill_missing_mode=fill_missing_mode,
        )
        for L in sorted(stats.keys()):
            rec = stats[L]
            rows.append({
                "method": name,
                "level": L,
                "num_communities": rec["num_communities"],
                "modularity": rec["modularity"],
                "top_5_community": json.dumps(rec["top5"]),
            })
    return rows


def write_modularity_csv(
    rows: List[Dict[str, Any]],
    csv_out: str = "modularity_by_level.csv",
    *,
    overwrite: bool = True,
):
    header = ["method", "level", "num_communities", "modularity", "top_5_community"]
    os.makedirs(os.path.dirname(csv_out) or ".", exist_ok=True)
    mode = "w" if overwrite or (not os.path.exists(csv_out)) else "a"
    write_header = (mode == "w")
    with open(csv_out, mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if write_header:
            w.writeheader()
        for r in rows:
            # round modularity for readability; keep full precision if you prefer
            out = dict(r)
            if isinstance(out.get("modularity"), float):
                out["modularity"] = round(out["modularity"], 6)
            w.writerow(out)
