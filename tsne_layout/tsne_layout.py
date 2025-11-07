from sklearn.manifold import TSNE
import numpy as np
from models.network_module_2 import LawNetwork

from node2vec import Node2Vec
from sklearn.preprocessing import OneHotEncoder


class TSNELayout:
    def __init__(self, law_graph, extent=None):

        self.network = LawNetwork(law_graph)
        self.community_graph = self.network.getCommunityGraph()


    def tsn_layout(self):

        G = self.network.graph

        # Detect communities using the Louvain method
        communities = self.network.getCommunities()

        # Convert the list of sets into a dictionary for easier node lookup
        community_dict = {}
        for id in communities:
            community = communities[id]
            for node in community.nodes:
                community_dict[node] = id
        

        # Generate node embeddings using node2vec
        node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=1)
        model = node2vec.fit(window=10, min_count=1, batch_words=4)
        node_embeddings = np.array([model.wv[str(node)] for node in G.nodes()])

        # One-hot encode the community information
        community_numbers = np.array([community_dict[node] for node in G.nodes()]).reshape(-1, 1)
        encoder = OneHotEncoder(sparse_output=False)
        community_onehot = encoder.fit_transform(community_numbers)

        # Combine node embeddings with one-hot encoded community info
        node_features = np.hstack((node_embeddings, community_onehot))

        # Apply t-SNE to the combined features
        tsne = TSNE(n_components=2, random_state=42, perplexity=50)
        node_positions = tsne.fit_transform(node_features)

        # Normalize the coordinates to the range [0, 1]
        min_x, min_y = np.min(node_positions, axis=0)
        max_x, max_y = np.max(node_positions, axis=0)
        node_positions = (node_positions - [min_x, min_y]) / (max_x - min_x, max_y - min_y)

        # self.network.find_central_points()

        node_id_to_index = {node: idx for idx, node in enumerate(self.network.graph.nodes())}
        positions = {node: (float(node_positions[node_id_to_index[node], 0]), float(node_positions[node_id_to_index[node], 1])) for node in self.network.graph.nodes()}

        return community_dict, positions
    


        











