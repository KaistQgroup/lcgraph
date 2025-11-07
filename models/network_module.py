import numpy as np
from networkx.algorithms.community import louvain_communities
from networkx.algorithms.centrality import katz_centrality
import networkx as nx
from utils.utils import safe_str
from models.community_module import Community
from models.community_set_module import CommunitySet
from models.frame_module import Frame
import copy



class LayeredLawNetwork():

    def __init__(self, graph):

        self.graph = graph
        self.max_level = 2  # This defines the maximum number of levels in the network
        self.levels = {}  # Dictionary to store communities at each level
        self.all_communities = {}  # All communities for convenience

        self.detect_communities()  # Method to detect and create communities
        self.compute_degree_centrality()


    def detect_communities(self):
        # Using Louvain method to detect communities in the input graph
        communities = louvain_communities(self.graph, seed=123) #list of sets of each community nodes
        print(f"Number of communities: {len(communities)}")

        level = 1
        community_set = CommunitySet("0_0_0", communities, level, None, self.graph)
        community_set.get_community_graph()
        self.levels[level] = [community_set]  # Store the communities at the current level
        self.all_communities.update(community_set.get_communities_dict())

        self.detect_subcommunities(levels=self.max_level)
           

    def detect_subcommunities(self, levels=3):
        """Recursively create subcommunities up to the specified level."""

        def recursive_community_detection(nodes, current_level, current_community, upper_community, max_level):
            if current_level + 1 > max_level:
                return None

            subgraph = self.graph.subgraph(nodes)
            if subgraph.number_of_nodes() <= 1:
                return None
            
            communities = louvain_communities(subgraph, seed=123)
            uid = safe_str(current_level) + "_" + safe_str(upper_community) + "_" + safe_str(current_community.id)
            community_set = CommunitySet(uid, communities, current_level + 1, current_community, self.graph)
            self.all_communities.update(community_set.get_communities_dict())

            if current_level + 1 in self.levels:
                self.levels[current_level + 1].append(community_set)
            else:
                self.levels[current_level + 1] = [community_set]
            
            community_objs = []
            for i, sub_community in community_set.get_communities_dict().items():
                sub_community.sub_communities = recursive_community_detection(sub_community.nodes, current_level + 1, sub_community, current_community.id, max_level)
                community_objs.append(sub_community)

            return community_objs

        # Start recursive community detection for the base level communities
        
        for community in self.levels[1][0].get_communities_dict().values():
            community.sub_communities = recursive_community_detection(community.nodes, 1, community, None, levels)
    

    def compute_degree_centrality(self):
        """Compute degree centrality for each node in the original graph."""
        try:
            self.degree_centralities = nx.degree_centrality(self.graph)
        except nx.exception.PowerIterationFailedConvergence:
            print(f"Degree centrality failed")
            raise
    



# import numpy as np
# import networkx as nx
# from networkx.algorithms.centrality import katz_centrality
# from utils.utils import safe_str
# from models.community_module import Community
# from models.community_set_module import CommunitySet
# from models.frame_module import Frame
# from typing import Optional


# class LayeredLawNetwork:
#     def __init__(self, graph: nx.Graph, algo: str = "louvain"):
#         self.graph = graph
#         self.levels = {}
#         self.all_communities = {}
#         self.algo = algo.lower()

#         # internal defaults
#         self._seed = 123
#         self._objective = "modularity"  # used only for Leiden
#         self._resolution = 1.0          # used only for Leiden CPM
#         self.max_level = 2
#         self._hsbm_degree_corrected = True
#         self._hsbm_use_finest = True    # set False to take coarsest level for level 1


#         self._ls_auto = True
#         self._ls_centers = None      # set an int to force a specific scale
#         self._ls_maximum_tree = True
#         self._last_ls_centers = None

#         # self.compare_methods_inplace(methods=("louvain","leiden","hsbm","ls"), levels=(1,2, 3))

#         self.detect_communities()
#         self.compute_degree_centrality()

#     def _run_community_detection(self, G: nx.Graph, nodes_subset=None):
#         H = G if nodes_subset is None else G.subgraph(nodes_subset).copy()
#         if self.algo == "louvain":
#             from networkx.algorithms.community import louvain_communities
#             return louvain_communities(H, seed=self._seed)
#         elif self.algo == "leiden":
#             from utils.leiden_helpers import leiden_communities_nx
#             return leiden_communities_nx(H, seed=self._seed, objective="CPM")
#         elif self.algo == "hsbm":
#             from utils.hsbm_inference import run_hsbm, extract_level_partitions
#             state, node_of = run_hsbm(H, degree_corrected=True)
#             return extract_level_partitions(state, node_of)[-1]
#         elif self.algo == "ls":
#             from utils.ls_helpers import ls_communities_nx
#             comms, centers = ls_communities_nx(
#                 H,
#                 center_num=self._ls_centers,        # None for auto
#                 auto_choose_centers=self._ls_auto,  # True by default
#                 maximum_tree=self._ls_maximum_tree, # True by default
#                 seed=self._seed,
#             )
#             self._last_ls_centers = centers       # keep for visualization
#             return comms
#         else:
#             raise ValueError("algo must be 'louvain', 'leiden', 'hsbm', or 'ls'")


#     def detect_communities(self):
#         communities = self._run_community_detection(self.graph)
#         print(f"[Level 1] Number of communities ({self.algo}): {len(communities)}")

#         level = 1
#         community_set = CommunitySet("0_0_0", communities, level, None, self.graph)
#         community_set.get_community_graph()
#         self.levels[level] = [community_set]
#         self.all_communities.update(community_set.get_communities_dict())

#         self.detect_subcommunities(levels=self.max_level)

#     def detect_subcommunities(self, levels=3):
#         def rec(nodes, current_level, current_community, upper_community, max_level):
#             if current_level + 1 > max_level:
#                 return None
#             subgraph = self.graph.subgraph(nodes).copy()
#             if subgraph.number_of_nodes() <= 1:
#                 return None

#             communities = self._run_community_detection(subgraph)

#             uid = f"{current_level}_{upper_community}_{current_community.id}"
#             community_set = CommunitySet(uid, communities, current_level + 1, current_community, self.graph)
#             self.all_communities.update(community_set.get_communities_dict())
#             self.levels.setdefault(current_level + 1, []).append(community_set)

#             objs = []
#             for _, sub_comm in community_set.get_communities_dict().items():
#                 sub_comm.sub_communities = rec(
#                     sub_comm.nodes, current_level + 1, sub_comm,
#                     current_community.id, max_level
#                 )
#                 objs.append(sub_comm)
#             return objs

#         for community in self.levels[1][0].get_communities_dict().values():
#             community.sub_communities = rec(community.nodes, 1, community, None, levels)

#     def compute_degree_centrality(self):
#         try:
#             self.degree_centralities = nx.degree_centrality(self.graph)
#         except nx.exception.PowerIterationFailedConvergence:
#             print("Degree centrality failed")
#             raise
    
#     def compare_methods_inplace(
#         self,
#         methods=("louvain", "leiden", "hsbm", "ls"),
#         levels=(1,),
#         ls_forced_k: Optional[int] = None,
#         pretty: bool = True,
#     ):

#         """
#         In-place comparison: temporarily switch self.algo (and LS params), rebuild communities,
#         collect metrics for the requested levels, then restore the original state.
#         Reports: Modularity, Top-5 community sizes, Runtime (seconds).
#         Returns: List[dict] rows.
#         """
#         import time
#         from utils.compare_helpers import (
#             extract_partition_at_level as _extract,
#             modularity as _mod,
#             top5_sizes as _top5,
#             pretty_print as _pp,
#         )

#         # Snapshot original state to restore later
#         orig = {
#             "algo": self.algo,
#             "levels": self.levels,
#             "all_communities": self.all_communities,
#             "ls_auto": getattr(self, "_ls_auto", None),
#             "ls_centers": getattr(self, "_ls_centers", None),
#             "ls_maximum_tree": getattr(self, "_ls_maximum_tree", None),
#             "last_ls_centers": getattr(self, "_last_ls_centers", None),
#             "hsbm_degree_corrected": getattr(self, "_hsbm_degree_corrected", None),
#             "hsbm_use_finest": getattr(self, "_hsbm_use_finest", None),
#             "max_level": getattr(self, "max_level", None),
#             "seed": getattr(self, "_seed", None),
#             "objective": getattr(self, "_objective", None),
#             "resolution": getattr(self, "_resolution", None),
#         }

#         rows = []
#         try:
#             built = {}

#             # Optionally include LS at a forced k as an extra "method label"
#             method_list = list(methods)
#             if ls_forced_k is not None and "ls" in method_list:
#                 method_list.append(f"ls(k={int(ls_forced_k)})")

#             for method in method_list:
#                 # Configure method-specific params
#                 forced_ls = method.startswith("ls(k=")
#                 algo_name = "ls" if forced_ls else method

#                 # Reset containers before each run
#                 self.algo = algo_name
#                 self.levels = {}
#                 self.all_communities = {}
#                 # Configure LS scale if needed
#                 if algo_name == "ls":
#                     if forced_ls:
#                         self._ls_auto = False
#                         self._ls_centers = int(method.split("=")[1].rstrip(")"))
#                     else:
#                         self._ls_auto = orig["ls_auto"]
#                         self._ls_centers = orig["ls_centers"]
#                     # keep maximum_tree as original
#                     self._ls_maximum_tree = orig["ls_maximum_tree"]

#                 # Re-run detection and time it
#                 t0 = time.perf_counter()
#                 self.detect_communities()
#                 elapsed = time.perf_counter() - t0

#                 # Collect metrics for requested levels
#                 for lvl in levels:
#                     comms = _extract(self, int(lvl))
#                     if not comms:
#                         rows.append({
#                             "method": method,
#                             "level": int(lvl),
#                             "k": 0,
#                             "modularity": None,
#                             "runtime_sec": elapsed,
#                             "top5_sizes": [],
#                             "note": f"level {int(lvl)} not available",
#                         })
#                         continue
#                     try:
#                         mod = _mod(self.graph, comms)
#                     except Exception:
#                         mod = None

#                     rows.append({
#                         "method": method,
#                         "level": int(lvl),
#                         "k": len(comms),
#                         "modularity": mod,
#                         "runtime_sec": elapsed,
#                         "top5_sizes": _top5(comms),
#                     })

#                 built[method] = True

#         finally:
#             # Restore original state
#             self.algo = orig["algo"]
#             self.levels = orig["levels"]
#             self.all_communities = orig["all_communities"]
#             self._ls_auto = orig["ls_auto"]
#             self._ls_centers = orig["ls_centers"]
#             self._ls_maximum_tree = orig["ls_maximum_tree"]
#             self._last_ls_centers = orig["last_ls_centers"]
#             self._hsbm_degree_corrected = orig["hsbm_degree_corrected"]
#             self._hsbm_use_finest = orig["hsbm_use_finest"]
#             self.max_level = orig["max_level"]
#             self._seed = orig["seed"]
#             self._objective = orig["objective"]
#             self._resolution = orig["resolution"]

#         if pretty:
#             _pp(rows, title=f"In-place comparison (levels={list(levels)})")
#         return rows


