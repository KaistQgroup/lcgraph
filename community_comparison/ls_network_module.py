from models.community_module import Community
from models.community_set_module import CommunitySet
from utils.utils import safe_str
from typing import Any, Dict, List, Set, Optional, Tuple
import networkx as nx
from ls_helpers import ls_partition, ls_multilevel



class LSLayeredLawNetwork:


    def __init__(
        self,
        graph: nx.Graph,
        max_level: int = 5,
        min_comm_size: int = 10,
        level0_center_num: Optional[int] = None,
        level0_auto_choose_centers: bool = True,
        deep_center_num: Optional[int] = 8,       # e.g., 8 for finer splits
        deep_auto_choose_centers: bool = False,       # set False to force K at deep levels
        maximum_tree: bool = True,
        seed: int = 1,
        debug: bool = False,
    ):
        self.graph = graph
        self.max_level = max_level
        self.min_comm_size = min_comm_size

        self.level0_center_num = level0_center_num
        self.level0_auto_choose_centers = level0_auto_choose_centers
        self.deep_center_num = deep_center_num
        self.deep_auto_choose_centers = deep_auto_choose_centers
        self.maximum_tree = maximum_tree
        self.seed = seed
        self.debug = debug

        self.levels: Dict[int, List[CommunitySet]] = {}
        self.all_communities: Dict[str, Any] = {}

        self.detect_communities()

    def detect_communities(self):
        levels_ls: List[List[Set[Any]]] = ls_multilevel(
            self.graph,
            max_levels=self.max_level,
            level0_center_num=self.level0_center_num,
            level0_auto_choose_centers=self.level0_auto_choose_centers,
            deep_center_num=self.deep_center_num,
            deep_auto_choose_centers=self.deep_auto_choose_centers,
            maximum_tree=self.maximum_tree,
            seed=self.seed,
            debug=self.debug,
        )

        # print("len(G) =", self.graph.number_of_nodes(), "len(E) =", self.graph.number_of_edges())

        if not levels_ls or not levels_ls[0]:
            root_comms = [set(self.graph.nodes())]
            root_cs = CommunitySet("0_0_0", root_comms, 1, None, self.graph)
            root_cs.get_community_graph()
            self.levels[1] = [root_cs]
            self.all_communities.update(root_cs.get_communities_dict())
            # print("levels keys:", list(self.levels.keys()))
            # for lvl, comms in self.levels.items():
            #     print(f"level {lvl}: {len(comms)} communities")
            return

        # ---- Level 1 ----
        root_comms: List[Set[Any]] = levels_ls[0]
        root_cs = CommunitySet("0_0_0", root_comms, 1, None, self.graph)
        root_cs.get_community_graph()
        self.levels[1] = [root_cs]
        self.all_communities.update(root_cs.get_communities_dict())

        parent_map_level1: Dict[Any, str] = {}
        for cid, comm in root_cs.get_communities_dict().items():
            for u in comm.nodes:
                parent_map_level1[u] = cid

        # ---- Deeper levels ----
        max_depth_available = min(len(levels_ls) - 1, self.max_level - 1)
        for depth in range(1, max_depth_available + 1):
            project_level = depth + 1
            comms_depth: List[Set[Any]] = levels_ls[depth]
            if not comms_depth:
                continue

            # choose parent map
            if depth == 1:
                parent_map = parent_map_level1
            else:
                parent_map: Dict[Any, str] = {}
                for pcs in self.levels[project_level - 1]:
                    for pid, pcomm in pcs.get_communities_dict().items():
                        for u in pcomm.nodes:
                            parent_map[u] = pid

            grouped_children: Dict[str, List[Set[Any]]] = {}
            for child_set in comms_depth:
                if not child_set:
                    continue
                any_node = next(iter(child_set))
                parent_id = parent_map.get(any_node)
                if parent_id is None and parent_map_level1:
                    parent_id = next(iter(parent_map_level1.values()))
                if parent_id is None:
                    continue
                grouped_children.setdefault(parent_id, []).append(child_set)

            current_level_sets: List[CommunitySet] = []
            parent_sets: List[CommunitySet] = self.levels.get(project_level - 1, [])
            for pcs in parent_sets:
                for pid, pcomm in pcs.get_communities_dict().items():
                    child_list = grouped_children.get(pid, [])
                    if not child_list:
                        continue
                    uid = (
                        safe_str(project_level - 1)
                        + "_"
                        + safe_str(getattr(pcomm, "upper_level", "0"))
                        + "_"
                        + safe_str(pcomm.id)
                    )
                    cs = CommunitySet(uid, child_list, project_level, pcomm, self.graph)
                    cs.get_community_graph()
                    self.all_communities.update(cs.get_communities_dict())
                    current_level_sets.append(cs)

            if current_level_sets:
                self.levels[project_level] = current_level_sets

        # print("levels keys:", list(self.levels.keys()))
        # for lvl, comms in self.levels.items():
        #     print(f"level {lvl}: {len(comms)} communities")
