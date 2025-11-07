import networkx as nx

from sklearn.manifold import TSNE
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from models.network_module_2 import LawNetwork
import json



class SpringLayout:
    def __init__(self, law_graph, extent=None):

        self.network = LawNetwork(law_graph)

        self.community_graph = self.network.getCommunityGraph()
        
    def spring_law_layout(self):
        # Convert the list of sets into a dictionary for easier node lookup
        community_dict = {}
        for id in self.network.communities:
            community = self.network.communities[id]
            for node in community.nodes:
                community_dict[node] = id
        
        print("Communities were detected")

        # Initialize positions of nodes based on community
        initial_pos = {}
        for id in self.network.communities:
            angle = 2 * np.pi * id / len(self.network.communities)
            center = np.array([np.cos(angle), np.sin(angle)])
            community = self.network.communities[id]
            for node in community.nodes:
                initial_pos[node] = center + 0.1 * np.random.randn(2)

        print("initial positions were calculated")

        # Apply the spring layout with the initial positions
        positions = nx.spring_layout(self.network.graph, pos=initial_pos, k=0.1, iterations=100)


        return community_dict, positions
    
    def spring_article_layout(self, graph):
        # Generate spring layout positions
        positions = nx.spring_layout(graph)

        # Convert positions to JSON-friendly format
        positions_json = {str(node): {"x": pos[0], "y": pos[1]} for node, pos in positions.items()}

        # Write to JSON file
        with open("spring_layout_positions.json", "w") as json_file:
            json.dump(positions_json, json_file, indent=4)

        return positions_json
 

        











