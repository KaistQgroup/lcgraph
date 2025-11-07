# modularity_eval.py
import os, csv, networkx as nx
from typing import Dict, List, Set, Any, Optional, Tuple
from networkx.algorithms.community.quality import modularity as nx_modularity

# ========= SHARED HELPERS (DO NOT EDIT PER METHOD) =========

def _make_partition_total(
    G: nx.Graph,
    raw_groups,
    deduplicate: bool = True,
    fill_missing_as_singletons: bool = True,
) -> List[Set[Any]]:
    """Turn possibly overlapping/duplicated groups into a full disjoint partition."""
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

    if fill_missing_as_singletons:
        for u in all_nodes - covered:
            parts.append({u})

    # sanity
    assert set().union(*parts) == all_nodes, "Partition does not cover all nodes."
    return parts


def partition_from_level_obj(net, level: int, G: nx.Graph) -> List[Set[Any]]:
    """Extract communities from net.levels[level] and make it a total partition."""
    raw = []
    if hasattr(net, "levels") and (level in net.levels):
        for cs in net.levels[level]:
            # assumes CommunitySet.get_communities_dict() with objects that have .nodes
            for _, comm in cs.get_communities_dict().items():
                raw.append(comm.nodes)
    return _make_partition_total(G, raw, deduplicate=True, fill_missing_as_singletons=True)


def modularity_for_partition(
    G: nx.Graph,
    communities: List[Set[Any]],
    weight: Optional[str] = "auto",
    resolution: float = 1.0,
) -> float:
    # weight="auto" → use "weight" if present, else None
    if weight == "auto":
        weight_attr = None
        if G.number_of_edges() > 0:
            u, v, d = next(iter(G.edges(data=True)))
            weight_attr = "weight" if ("weight" in d) else None
    else:
        weight_attr = weight
    H = nx.Graph(G) if G.is_directed() else G
    return nx_modularity(H, communities, weight=weight_attr, resolution=resolution)


def modularity_all_levels(
    G: nx.Graph,
    net,
    weight: Optional[str] = "auto",
    resolution: float = 1.0,
    max_level: Optional[int] = None,
) -> Dict[int, Dict[str, Any]]:
    """Return {level: {'num_communities': int, 'modularity': float}}."""
    present_levels = sorted(getattr(net, "levels", {}).keys())
    if max_level is not None:
        present_levels = [L for L in present_levels if L <= max_level]

    out: Dict[int, Dict[str, Any]] = {}
    for L in present_levels:
        comms = partition_from_level_obj(net, L, G)
        q = modularity_for_partition(G, comms, weight=weight, resolution=resolution)
        out[L] = {"num_communities": len(comms), "modularity": q}
    return out

# ================= ONE RUNNER PER METHOD ===================
# (adjust the import paths to your project)

# 1) Louvain (recursive driver)
def run_louvain_recursive(
    G: nx.Graph,
    *,
    max_level: int = 3,
    weight: Optional[str] = "auto",
    resolution: float = 1.0,
    **kwargs,  # any extra ctor args your class supports
) -> Tuple[str, Dict[int, Dict[str, Any]]]:
    from models.network_module import LayeredLawNetwork
    net = LayeredLawNetwork(G)
    stats = modularity_all_levels(G, net, weight=weight, resolution=resolution, max_level=max_level)
    return "Louvain_recursive", stats


# 2) Hierarchical Louvain (contractive)
def run_hlouvain_contractive(
    G: nx.Graph,
    *,
    max_level: int = 3,
    weight: Optional[str] = "auto",
    resolution: float = 1.0,
    # typical ctor args for your HLouvainLayeredLawNetwork, if any:
    **kwargs,
) -> Tuple[str, Dict[int, Dict[str, Any]]]:
    from h_louvain_network_module import HLouvainLayeredLawNetwork
    net = HLouvainLayeredLawNetwork(G)
    stats = modularity_all_levels(G, net, weight=weight, resolution=resolution, max_level=max_level)
    return "Louvain_hier_contractive", stats


# 3) HSBM (multilevel)
def run_hsbm(
    G: nx.Graph,
    *,
    max_level: int = 3,
    weight: Optional[str] = "auto",
    resolution: float = 1.0,
    # pass model-specific args here if needed:
    **kwargs,
) -> Tuple[str, Dict[int, Dict[str, Any]]]:
    from hsbm_network_module import HSBMLayeredLawNetwork
    net = HSBMLayeredLawNetwork(G)
    stats = modularity_all_levels(G, net, weight=weight, resolution=resolution, max_level=max_level)
    return "HSBM_multi", stats


# 4) Local Search (LS) multilevel
def run_ls_multilevel(
    G: nx.Graph,
    *,
    max_level: int = 3,
    weight: Optional[str] = "auto",
    resolution: float = 1.0,
    # LS-specific knobs (match what you exposed in LSLayeredLawNetwork)
    level0_center_num: Optional[int] = None,
    level0_auto_choose_centers: bool = True,
    deep_center_num: Optional[int] = None,
    deep_auto_choose_centers: bool = True,
    maximum_tree: bool = True,
    seed: int = 1,
    debug: bool = False,
    # anything else your constructor expects:
    **kwargs,
) -> Tuple[str, Dict[int, Dict[str, Any]]]:
    from ls_network_module import LSLayeredLawNetwork
    net = LSLayeredLawNetwork(
        G,
        max_level=max_level,
        level0_center_num=level0_center_num,
        level0_auto_choose_centers=level0_auto_choose_centers,
        deep_center_num=deep_center_num,
        deep_auto_choose_centers=deep_auto_choose_centers,
        maximum_tree=maximum_tree,
        seed=seed,
        debug=debug,
        **kwargs,
    )
    stats = modularity_all_levels(G, net, weight=weight, resolution=resolution, max_level=max_level)
    return "ls_multi", stats

# ===================== ORCHESTRATOR ========================

def write_modularity_long_csv(
    G: nx.Graph,
    csv_out: str = "profiles/modularity_long.csv",
    *,
    # global quality params
    weight: Optional[str] = "auto",
    resolution: float = 1.0,
    # per-method params:
    louvain_recursive_params: Optional[dict] = None,
    hlouvain_params: Optional[dict] = None,
    hsbm_params: Optional[dict] = None,
    ls_params: Optional[dict] = None,
):
    """
    Calls each method with its own parameter dict and writes one long CSV:
      columns = [module, level, num_communities, modularity]
    """
    louvain_recursive_params = louvain_recursive_params or {}
    hlouvain_params = hlouvain_params or {}
    hsbm_params = hsbm_params or {}
    ls_params = ls_params or {}

    runners = [
        run_louvain_recursive,
        run_hlouvain_contractive,
        run_hsbm,
        run_ls_multilevel,
    ]

    os.makedirs(os.path.dirname(csv_out) or ".", exist_ok=True)
    newfile = not os.path.exists(csv_out)
    header = ["module", "level", "num_communities", "modularity"]

    with open(csv_out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if newfile:
            w.writeheader()

        for runner, params in zip(runners, [louvain_recursive_params, hlouvain_params, hsbm_params, ls_params]):
            label, stats_by_level = runner(G, weight=weight, resolution=resolution, **params)
            for L in sorted(stats_by_level.keys()):
                w.writerow({
                    "module": label,
                    "level": L,
                    "num_communities": stats_by_level[L]["num_communities"],
                    "modularity": round(stats_by_level[L]["modularity"], 6),
                })

    print(f"[OK] Wrote long-format modularity CSV → {csv_out}")
