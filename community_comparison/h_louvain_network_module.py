import networkx as nx
from typing import Dict, List, Set, Any
from utils.utils import safe_str
from models.community_set_module import CommunitySet
from louvain_hierarchy import louvain_multi_level

class HLouvainLayeredLawNetwork:

    def __init__(self, graph):
        self.graph = graph
        self.max_level = 2
        self.levels: Dict[int, List[CommunitySet]] = {}
        self.all_communities = {}
        self.detect_communities()

    def detect_communities(self):
        levels_l = louvain_multi_level(self.graph, max_depth=self.max_level, seed=123, resolution=1.0)

        print("len(G) =", self.graph.number_of_nodes(), "len(E) =", self.graph.number_of_edges())
        print("levels keys:", list(levels_l.keys()))
        for lvl, comms in self.levels.items():
            print(f"level {lvl}: {len(comms)} communities")


        if not levels_l:
            root_comms = [set(self.graph.nodes())]
            root_cs = CommunitySet("0_0_0", root_comms, 1, None, self.graph)
            root_cs.get_community_graph()
            self.levels[1] = [root_cs]
            self.all_communities.update(root_cs.get_communities_dict())
            return

        root_comms: List[Set[Any]] = levels_l.get(0, [set(self.graph.nodes())])
        root_cs = CommunitySet("0_0_0", root_comms, 1, None, self.graph)
        root_cs.get_community_graph()
        self.levels[1] = [root_cs]
        self.all_communities.update(root_cs.get_communities_dict())

        parent_map_level1: Dict[Any, str] = {}
        for cid, comm in root_cs.get_communities_dict().items():
            for u in comm.nodes:
                parent_map_level1[u] = cid

        max_louvain_level = max(levels_l.keys())
        for L in range(1, min(max_louvain_level, self.max_level - 1) + 1):
            project_level = L + 1
            parent_sets: List[CommunitySet] = self.levels[project_level - 1]
            comms_L: List[Set[Any]] = levels_l.get(L, [])
            if not comms_L:
                continue

            if L == 1:
                parent_map = parent_map_level1
            else:
                parent_map = {}
                for pcs in self.levels[project_level - 1]:
                    for pid, pcomm in pcs.get_communities_dict().items():
                        for u in pcomm.nodes:
                            parent_map[u] = pid

            grouped_children: Dict[str, List[Set[Any]]] = {}
            for child_set in comms_L:
                any_node = next(iter(child_set))
                parent_id = parent_map.get(any_node, None)
                if parent_id is None:
                    parent_id = next(iter(parent_map_level1.values()))
                grouped_children.setdefault(parent_id, []).append(child_set)

            current_level_sets: List[CommunitySet] = []
            for pcs in parent_sets:
                for pid, pcomm in pcs.get_communities_dict().items():
                    child_list = grouped_children.get(pid, [])
                    if not child_list:
                        continue
                    uid = safe_str(project_level - 1) + "_" + safe_str(
                        getattr(pcomm, "upper_level", "0")
                    ) + "_" + safe_str(pcomm.id)
                    cs = CommunitySet(uid, child_list, project_level, pcomm, self.graph)
                    self.all_communities.update(cs.get_communities_dict())
                    current_level_sets.append(cs)

            if current_level_sets:
                self.levels[project_level] = current_level_sets

