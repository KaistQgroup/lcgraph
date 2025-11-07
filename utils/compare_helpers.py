from typing import List, Set, Hashable, Dict, Any, Optional
import networkx as nx

def extract_partition_at_level(net: Any, level: int) -> Optional[List[Set[Hashable]]]:
    if level not in net.levels or not net.levels[level]:
        return None
    cs = net.levels[level][0]
    comms: List[Set[Hashable]] = []
    for _, c in cs.get_communities_dict().items():
        comms.append(set(c.nodes))
    return comms

def modularity(G: nx.Graph, comms: List[Set[Hashable]]) -> float:
    return nx.algorithms.community.quality.modularity(G, comms, weight="weight")

def top5_sizes(comms: List[Set[Hashable]]) -> List[int]:
    return sorted((len(c) for c in comms), reverse=True)[:5]

def pretty_print(rows: List[Dict], title: str = "") -> None:
    if title:
        print("\n" + title)
    header = f"{'Method':12} {'Lvl':>3} {'k':>5} {'Modularity':>12} {'Time(s)':>9} {'Top5 sizes':>22} {'Note':>10}"
    print(header)
    print("-" * len(header))
    for r in rows:
        mod = "-" if r.get("modularity") is None else f"{r['modularity']:.4f}"
        tms = "-" if r.get("runtime_sec") is None else f"{r['runtime_sec']:.2f}"
        note = r.get("note", "")
        print(f"{r['method']:12} {r['level']:>3d} {r['k']:>5d} {mod:>12} {tms:>9} {str(r['top5_sizes']):>22} {note:>10}")
