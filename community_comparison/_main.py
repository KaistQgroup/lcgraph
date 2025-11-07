import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from utils.lawdbcon import LawDB
# from models.network_module import LayeredLawNetwork


# law_db = LawDB(text_file = False, law_csv_file = False)
# law_graph = law_db.create_law_graph()

# network = LayeredLawNetwork(law_graph)




import time
from profiling import time_mem_section, write_csv_row, _rss_mb
from utils.lawdbcon import LawDB
from models.network_module import LayeredLawNetwork
from hsbm_network_module import HSBMLayeredLawNetwork
import hsbm_partition as HSBM
from h_louvain_network_module import HLouvainLayeredLawNetwork
import louvain_hierarchy as LH
from modularity_utils import write_modularity_long_csv
import os, csv, time
from typing import Dict, Any, List
from ls_network_module import LSLayeredLawNetwork

def profile_layered_louvain(
    dataset_name: str = "law_graph",
    max_level: int = 2,
    seed: int = 123,
    csv_out: str = "profiles/layered_louvain.csv",
):
    # 1) Build graph
    law_db = LawDB(text_file=False, law_csv_file=False)
    G = law_db.create_law_graph()
    n, m = G.number_of_nodes(), G.number_of_edges()

    # 2) Metrics container
    metrics = {
        "dataset": dataset_name,
        "method": "HierarchicalLouvain",
        "n": n, "m": m,
        "max_level": max_level, "seed": seed,
        "rss_start_mb": round(_rss_mb(), 3),
        "ts_start": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # 3) Save originals
    orig_detect_communities = LayeredLawNetwork.detect_communities
    orig_detect_subcommunities = LayeredLawNetwork.detect_subcommunities

    # 4) Wrap phase timings
    def wrapped_detect_communities(self):
        with time_mem_section(metrics, "phase_level1"):
            return orig_detect_communities(self)

    def wrapped_detect_subcommunities(self, levels=3):
        with time_mem_section(metrics, "phase_sublevels"):
            return orig_detect_subcommunities(self, levels=levels)

    LayeredLawNetwork.detect_communities = wrapped_detect_communities
    LayeredLawNetwork.detect_subcommunities = wrapped_detect_subcommunities

    # 5) Construct + total timing
    from profiling import time_mem_section  # re-use context manager
    with time_mem_section(metrics, "total"):
        net = LayeredLawNetwork(G)
        net.max_level = max_level  # ensure consistent with desired max level

    # 6) Restore originals
    LayeredLawNetwork.detect_communities = orig_detect_communities
    LayeredLawNetwork.detect_subcommunities = orig_detect_subcommunities

    # 7) Collect per-level community counts
    level_counts = {}
    for lvl, cs_list in net.levels.items():
        cnt = 0
        for cs in cs_list:
            cnt += len(cs.get_communities_dict())
        level_counts[f"level{lvl}_communities"] = cnt
    metrics.update(level_counts)

    metrics["rss_end_mb"] = round(_rss_mb(), 3)

    # 8) Write CSV
    header = [
        "dataset","method","n","m","max_level","seed","ts_start",
        "rss_start_mb",
        "total_sec","total_rss_mb_delta","total_py_peak_mb",
        "phase_level1_sec","phase_level1_rss_mb_delta","phase_level1_py_peak_mb",
        "phase_sublevels_sec","phase_sublevels_rss_mb_delta","phase_sublevels_py_peak_mb",
        "level1_communities","level2_communities","level3_communities",
        "rss_end_mb"
    ]
    write_csv_row(csv_out, metrics, header=header)
    print(f"[OK] Profile written -> {csv_out}")
    return net, metrics


def _write_csv_row_safe(path: str, row: Dict[str, Any], header: List[str]) -> None:
    """Write a CSV row, trimming any keys not present in header and filling missing ones."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    newfile = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if newfile:
            w.writeheader()
        row_trimmed = {k: row.get(k, "") for k in header}
        w.writerow(row_trimmed)

def profile_hsbm_multilevel(
    dataset_name: str = "law_graph",
    max_level: int = 4,
    seed: int = 123,
    csv_out: str = "profiles/hsbm_multilevel.csv",
    min_size: int = 20,
):
    law_db = LawDB(text_file=False, law_csv_file=False)
    G = law_db.create_law_graph()
    n, m = G.number_of_nodes(), G.number_of_edges()

    metrics: Dict[str, Any] = {
        "dataset": dataset_name,
        "method": "HSBM_multi",
        "n": n,
        "m": m,
        "max_level": max_level,
        "seed": seed,
        "min_size": min_size,
        "rss_start_mb": round(_rss_mb(), 3),
        "ts_start": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # --- monkey-patch to time ONLY the top-level fit ---
    _orig_hsbm_multi = HSBM.hsbm_multi_level

    def _wrapped_hsbm_multi(
        G_in,
        max_depth=max_level,          # use the function arg, not a hardcoded number
        min_size=min_size,
        seed=seed,
        binarize=True,
        candidates=(2, 3, 4),
    ):
        with time_mem_section(metrics, "phase_hsbm_fit"):
            return _orig_hsbm_multi(
                G_in,
                max_depth=max_depth,
                min_size=min_size,
                seed=seed,
                binarize=binarize,
                candidates=candidates,
            )

    HSBM.hsbm_multi_level = _wrapped_hsbm_multi
    # ---------------------------------------------------

    with time_mem_section(metrics, "total"):
        net = HSBMLayeredLawNetwork(G)
        # If your class reads max level internally, this line may be unnecessary.
        # Keeping it for clarity; the wrapper already enforces max_depth=max_level.
        net.max_level = max_level

    # restore original (important)
    HSBM.hsbm_multi_level = _orig_hsbm_multi

    # counts per level (whatever levels HSBM produced)
    level_counts: Dict[str, int] = {}
    for lvl, cs_list in net.levels.items():
        cnt = 0
        for cs in cs_list:
            cnt += len(cs.get_communities_dict())
        level_counts[f"level{lvl}_communities"] = cnt
    metrics.update(level_counts)

    metrics["rss_end_mb"] = round(_rss_mb(), 3)

    # ----- dynamic header that matches ACTUAL levels -----
    base_header = [
        "dataset", "method", "n", "m", "max_level", "seed", "min_size", "ts_start",
        "rss_start_mb",
        "total_sec", "total_rss_mb_delta", "total_py_peak_mb",
        "phase_hsbm_fit_sec", "phase_hsbm_fit_rss_mb_delta", "phase_hsbm_fit_py_peak_mb",
    ]
    present_levels = sorted(net.levels.keys())  # handles 0- or 1-based indices
    level_cols = [f"level{L}_communities" for L in present_levels]
    header = base_header + level_cols + ["rss_end_mb"]
    # -----------------------------------------------------

    _write_csv_row_safe(csv_out, metrics, header=header)
    print(f"[OK] Profile written -> {csv_out}")
    return net, metrics


def profile_hsbm_twolevels(
    dataset_name: str = "law_graph",
    seed: int = 123,
    csv_out: str = "profiles/hsbm_twolevels.csv",
    min_size: int = 20,
):

    law_db = LawDB(text_file=False, law_csv_file=False)
    G = law_db.create_law_graph()
    n, m = G.number_of_nodes(), G.number_of_edges()

    metrics = {
        "dataset": dataset_name,
        "method": "HSBM_twolevels",
        "n": n, "m": m,
        "seed": seed, "min_size": min_size, "max_depth": 4,
        "rss_start_mb": round(_rss_mb(), 3),
        "ts_start": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with time_mem_section(metrics, "total"):
        with time_mem_section(metrics, "phase_hsbm_fit"):
            levels = HSBM.hsbm_multi_level(G, max_depth=4, min_size=min_size, seed=seed)

    level1_count = len(levels.get(0, [set(G.nodes())]))
    level2_count = len(levels.get(1, []))
    metrics["level1_communities"] = level1_count
    metrics["level2_communities"] = level2_count
    metrics["rss_end_mb"] = round(_rss_mb(), 3)

    header = [
        "dataset","method","n","m","seed","min_size","max_depth","ts_start",
        "rss_start_mb",
        "total_sec","total_rss_mb_delta","total_py_peak_mb",
        "phase_hsbm_fit_sec","phase_hsbm_fit_rss_mb_delta","phase_hsbm_fit_py_peak_mb",
        "level1_communities","level2_communities",
        "rss_end_mb"
    ]
    write_csv_row(csv_out, metrics, header=header)
    print(f"[OK] Profile written -> {csv_out}")
    return levels, metrics

def profile_louvain_hier_multilevel(
    dataset_name: str = "law_graph",
    max_level: int = 2,
    seed: int = 123,
    resolution: float = 1.0,
    csv_out: str = "profiles/louvain_hier_multilevel.csv",
):
    law_db = LawDB(text_file=False, law_csv_file=False)
    G = law_db.create_law_graph()
    n, m = G.number_of_nodes(), G.number_of_edges()

    metrics = {
        "dataset": dataset_name,
        "method": "Louvain_hier_contractive",
        "n": n, "m": m,
        "max_level": max_level, "seed": seed, "resolution": resolution,
        "rss_start_mb": round(_rss_mb(), 3),
        "ts_start": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    _orig_louvain_multi = LH.louvain_multi_level
    def _wrapped_louvain_multi(G_in, max_depth=None, seed=seed, resolution=resolution, weight_attr="weight"):
        with time_mem_section(metrics, "phase_louvain_fit"):
            return _orig_louvain_multi(G_in, max_depth=max_depth, seed=seed, resolution=resolution, weight_attr=weight_attr)

    LH.louvain_multi_level = _wrapped_louvain_multi

    with time_mem_section(metrics, "total"):
        net = HLouvainLayeredLawNetwork(G)
        net.max_level = max_level  # just in case

    LH.louvain_multi_level = _orig_louvain_multi

    level_counts = {}
    for lvl, cs_list in net.levels.items():
        cnt = 0
        for cs in cs_list:
            cnt += len(cs.get_communities_dict())
        level_counts[f"level{lvl}_communities"] = cnt
    metrics.update(level_counts)

    metrics["rss_end_mb"] = round(_rss_mb(), 3)

    header = [
        "dataset","method","n","m","max_level","seed","resolution","ts_start",
        "rss_start_mb",
        "total_sec","total_rss_mb_delta","total_py_peak_mb",
        "phase_louvain_fit_sec","phase_louvain_fit_rss_mb_delta","phase_louvain_fit_py_peak_mb",
        "level1_communities","level2_communities","level3_communities",
        "rss_end_mb"
    ]
    write_csv_row(csv_out, metrics, header=header)
    print(f"[OK] Profile written -> {csv_out}")
    return net, metrics


def profile_louvain_hier_twolevels(
    dataset_name: str = "law_graph",
    seed: int = 123,
    resolution: float = 1.0,
    csv_out: str = "profiles/louvain_hier_twolevels.csv",
):
    from louvain_hierarchy import louvain_multi_level

    law_db = LawDB(text_file=False, law_csv_file=False)
    G = law_db.create_law_graph()
    n, m = G.number_of_nodes(), G.number_of_edges()

    metrics = {
        "dataset": dataset_name,
        "method": "Louvain_hier_twolevels",
        "n": n, "m": m,
        "seed": seed, "resolution": resolution, "max_depth": 2,
        "rss_start_mb": round(_rss_mb(), 3),
        "ts_start": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with time_mem_section(metrics, "total"):
        with time_mem_section(metrics, "phase_louvain_fit"):
            levels = louvain_multi_level(G, max_depth=2, seed=seed, resolution=resolution)

    level1_count = len(levels.get(0, [set(G.nodes())]))
    level2_count = len(levels.get(1, []))
    metrics["level1_communities"] = level1_count
    metrics["level2_communities"] = level2_count
    metrics["rss_end_mb"] = round(_rss_mb(), 3)

    header = [
        "dataset","method","n","m","seed","resolution","max_depth","ts_start",
        "rss_start_mb",
        "total_sec","total_rss_mb_delta","total_py_peak_mb",
        "phase_louvain_fit_sec","phase_louvain_fit_rss_mb_delta","phase_louvain_fit_py_peak_mb",
        "level1_communities","level2_communities",
        "rss_end_mb"
    ]
    write_csv_row(csv_out, metrics, header=header)
    print(f"[OK] Profile written -> {csv_out}")
    return levels, metrics



if __name__ == "__main__":
    # profile_hsbm_multilevel()
    # profile_hsbm_twolevels()
    # profile_layered_louvain()
    # profile_louvain_hier_multilevel()
    law_db = LawDB(text_file=True, law_csv_file=False)
    G = law_db.create_law_graph()

    hsbm_model = HSBMLayeredLawNetwork(G)
    hsbm_model.print_level_counts()

    # ls_model = LSLayeredLawNetwork(
    #     G,
    #     max_level=5,
    #     min_comm_size=5,
    #     deep_center_num=8,            # try 4/8/16…
    #     deep_auto_choose_centers=False,  # force K
    #     debug=True
    # )




    # print("Available levels:", list(ls_model.levels.keys()))
    # for lvl, cs_list in ls_model.levels.items():
    #     print(f"\nLevel {lvl}: {len(cs_list)} CommunitySet group(s)")
    #     for i, cs in enumerate(cs_list, 1):
    #         # Each CommunitySet represents the children for one parent at the previous level
    #         comms_dict = cs.get_communities_dict()  # {community_id: Community}
    #         print(f"  Group {i}: {len(comms_dict)} communities")
    #         # Optional: list community sizes
    #         sizes = [len(c.nodes) for c in comms_dict.values()]
    #         print("   sizes:", sorted(sizes, reverse=True)[:10], "...")





    # write_modularity_long_csv(
    #     G,
    #     csv_out="profiles/modularity_long_patents.csv",
    #     weight="auto",
    #     resolution=1.0,
    #     louvain_recursive_params={"max_level": 4},   # only what THIS method needs
    #     hlouvain_params={"max_level": 4},
    #     hsbm_params={"max_level": 4},
    #     ls_params={
    #         "max_level": 5,
    #         "level0_auto_choose_centers": True,
    #         "deep_auto_choose_centers": False,
    #         "deep_center_num": 8,
    #         "maximum_tree": True,
    #         "seed": 1,
    #         "debug": False,
    #     },)

    # print(results)


