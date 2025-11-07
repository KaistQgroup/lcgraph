from typing import Dict, List, Set, Hashable, Optional, Tuple
import networkx as nx

from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]     
LS_REPO_DIR  = PROJECT_ROOT / "external/ls"
if str(LS_REPO_DIR) not in sys.path:
    sys.path.insert(0, str(LS_REPO_DIR))
from LS_algorithm import hierarchical_degree_communities  # type: ignore


def _labels_to_partition(nodes: List[Hashable], labels) -> List[Set[Hashable]]:
    if isinstance(labels, dict):
        ordered = [labels[n] for n in nodes]
    else:
        ordered = list(labels)
        if len(ordered) != len(nodes):
            raise ValueError("Label vector length does not match number of nodes.")
    groups: Dict[int, Set[Hashable]] = {}
    for n, cid in zip(nodes, ordered):
        groups.setdefault(int(cid), set()).add(n)
    return list(groups.values())

def ls_partition(
    G: nx.Graph,
    center_num: Optional[int] = None,
    auto_choose_centers: bool = True,
    maximum_tree: bool = True,
    seed: int = 1,
) -> Tuple[List[Set[Hashable]], List[Hashable]]:
    """
    Single LS run -> one flat partition on G.
    Returns (communities, centers)
    """
    D, centers, y_ls, y_ls_partition, _ = hierarchical_degree_communities(
        G,
        0 if center_num is None else center_num,
        auto_choose_centers=auto_choose_centers,
        maximum_tree=maximum_tree,
        seed=seed,
    )
    comms = _labels_to_partition(list(G.nodes()), y_ls_partition)
    return comms, centers

def ls_multilevel(
    G: nx.Graph,
    *,
    max_levels: int = 2,
    # top-level selection policy (no limits applied to the result)
    level0_center_num: Optional[int] = None,     # e.g., set to an int to force K centers
    level0_auto_choose_centers: bool = True,     # True = use LS decision-graph gap
    # deep levels selection policy (again, no result filtering)
    deep_center_num: Optional[int] = None,       # e.g., 4/8/16 to force finer splits
    deep_auto_choose_centers: bool = True,       # True = auto gap at each subgraph
    maximum_tree: bool = True,
    seed: int = 1,
    debug: bool = False,
    # what to do if a subgraph has zero edges (LS would crash):
    # - "components": split into connected components (results in singletons if e==0)
    # - "keep": keep the whole parent as a single community
    edgeless_strategy: str = "components",
) -> List[List[Set[Hashable]]]:
    """
    Multi-level LS without any size filtering or 'keep-largest' fallbacks.
    Returns: levels[depth] -> list of communities (as sets of node ids), where
      depth=0 is the top-level partition on G, depth=1 are children inside each
      level-0 community, etc.

    Notes:
    - No thresholds are applied; all raw LS splits are kept.
    - Edgeless/trivial subgraphs are handled to avoid LS internal errors.
    - Use center_num/auto_choose flags to steer granularity per level.
    """
    levels: List[List[Set[Hashable]]] = []

    # ---------- Level 0 (top-level over full graph) ----------
    try:
        level0_comms, _ = ls_partition(
            G,
            center_num=(None if level0_auto_choose_centers else (0 if level0_center_num is None else level0_center_num)),
            auto_choose_centers=level0_auto_choose_centers,
            maximum_tree=maximum_tree,
            seed=seed,
        )
    except Exception as ex:
        # If LS fails at the very top (extremely unlikely), fall back to 1 community
        if debug:
            print(f"[L0] LS failed ({ex}); using 1 community with all nodes.")
        level0_comms = [set(G.nodes())]

    if debug:
        print(f"[L0] raw split: {len(level0_comms)} parts; sizes={[len(c) for c in level0_comms]}")
    # Keep ALL communities (no filtering)
    level0_kept = [set(c) for c in level0_comms]
    levels.append(level0_kept)

    # Prepare queue for deeper recursion
    current_level_graphs = [(G.subgraph(c).copy(), c) for c in level0_kept]

    # ---------- Deeper levels (1 .. max_levels-1) ----------
    for depth in range(1, max_levels):
        next_level: List[Set[Hashable]] = []
        new_graphs = []

        for subG, parent_nodes in current_level_graphs:
            n_sub = subG.number_of_nodes()
            e_sub = subG.number_of_edges()

            # Guard: trivial / edgeless subgraphs (avoid LS crash)
            if n_sub <= 1 or e_sub == 0:
                if edgeless_strategy == "components":
                    comps = [set(c) for c in nx.connected_components(subG.to_undirected())]
                    if debug:
                        print(f"[L{depth}] n={n_sub}, e={e_sub} -> components: {len(comps)}; sizes={[len(c) for c in comps]}")
                    kept = comps
                else:  # "keep"
                    if debug:
                        print(f"[L{depth}] n={n_sub}, e={e_sub} -> keep parent as single community")
                    kept = [set(subG.nodes())]

                # Append ALL (no filtering)
                next_level.extend(kept)
                for c in kept:
                    new_graphs.append((subG.subgraph(c).copy(), c))
                continue

            # Choose selection policy for deep levels
            if deep_auto_choose_centers:
                k_arg = None
                auto = True
            else:
                k_arg = 0 if deep_center_num is None else deep_center_num
                auto = False

            # Run LS on the subgraph
            try:
                comms, _ = ls_partition(
                    subG,
                    center_num=k_arg,
                    auto_choose_centers=auto,
                    maximum_tree=maximum_tree,
                    seed=seed + depth,
                )
            except Exception as ex:
                if debug:
                    print(f"[L{depth}] LS failed on parent_size={len(parent_nodes)} ({ex}); keep parent intact.")
                comms = [set(subG.nodes())]

            if debug:
                print(f"[L{depth}] parent_size={len(parent_nodes)} -> raw split: {len(comms)}; sizes={[len(c) for c in comms]}")

            # Keep ALL raw parts (no min size, no keep-largest)
            kept = [set(c) for c in comms] if comms else [set(subG.nodes())]

            next_level.extend(kept)
            for c in kept:
                new_graphs.append((subG.subgraph(c).copy(), c))

        if not next_level:
            break

        levels.append(next_level)
        current_level_graphs = new_graphs

    return levels
