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
    

