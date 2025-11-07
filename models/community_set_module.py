from models.frame_module import Frame
from models.community_module import Community
from utils.utils import safe_str
from models.community_graph_module import CommunityGraph

class CommunitySet():

    def __init__(self, id, communities, level, upper_community, graph) -> None:
        self.id = id
        self.upper_community = upper_community
        self.level = level
        self.communities = communities
        self.frame = None
        self.disconnected_communities = None
        self.G_community = None
        self.upper_community_id = None
        self.graph = graph

        if self.upper_community is not None:
            self.upper_community_id = self.upper_community.id
        self.communities_dict = self.set_communities_dict()
    
    def get_frame(self):
        if self.frame is None:
            self.frame = Frame(communities=self.get_communities_dict(only_connected=True), hexa_grid = True)

            if self.id == "0_0_0":
                self.frame.extract_layer_disconnected_grids(len(self.get_disconnected_communities()))
            else:
                self.set_scale_factors()
        
        return self.frame

    def get_spring_frame(self):

        if self.frame is None:
            if self.id == "0_0_0":
                g = self.get_community_graph(only_connected=True)
                self.frame = Frame(self.get_communities_dict(only_connected=True), spring_grids= True, graph=g)
                self.frame.extract_layer_disconnected_grids(len(self.get_disconnected_communities()))
            else:
                g = self.get_community_graph()
                self.frame = Frame(self.get_communities_dict(), spring_grids= True, graph=g.g_c)
                self.set_scale_factors()
        
        return self.frame


    def get_disconnected_communities(self):
        if self.disconnected_communities == None:
            self.disconnected_communities = []
            for key, comm in self.communities_dict.items():
                if comm.get_is_disconnected():
                    self.disconnected_communities.append(comm)
           
            return self.disconnected_communities
        else:
            return self.disconnected_communities
    

    def get_community_graph(self, only_connected = False):
        if self.G_community is None:
            self.G_community = CommunityGraph(self.graph, self.communities_dict)
            self.detect_disconnected_communities()

        if self.id == "0_0_0" and only_connected:
            connected = [key for key, com in self.communities_dict.items() if key not in self.disconnected_communities]
            connected_graph = self.G_community.g_c.subgraph(connected)
            return connected_graph

        
        return self.G_community
    
    def get_communities_dict(self, only_connected=False):
        if self.id == "0_0_0" and only_connected:
            return {key: self.communities_dict[key] for key in self.communities_dict if key not in self.disconnected_communities}

        return self.communities_dict

    def detect_disconnected_communities(self):
        """Detects and stores disconnected communities."""
        self.disconnected_communities = self.G_community.get_disconnected_components()
        # if(len(self.disconnected_communities) > 0):
        #     print(f"Disconnected communities: {self.disconnected_communities}")
        
        for id, community in self.communities_dict.items():
            if id in self.disconnected_communities:
                community.set_is_disconnected(True)
            else:
                community.set_is_disconnected(False)
    
    def set_communities_dict(self):
        c_d = {}
        for i, nodes in enumerate(self.communities):
            c = Community(i, nodes, self.level, self.upper_community_id, upper_level= self.level -1)
            c_d.update({c.uId: c})
        return c_d
    
    def set_community_positions(self, community_positions, add_disconnected = False):
        positions = community_positions
        if self.id == "0_0_0" and add_disconnected:
            positions.update({key: self.community_positions[key] for key in self.community_positions if key in self.disconnected_communities})
        self.community_positions = positions
    
    def get_community_positions(self, only_connected = True):
        if self.id == "0_0_0" and only_connected:
            return {key: self.community_positions[key] for key in self.community_positions if key not in self.disconnected_communities}
        return self.community_positions        
        

    
    def set_scale_factors(self):
        parent_central_position = self.upper_community.get_central_position()
        parent_radius = self.upper_community.radius

        parent_x_min = parent_central_position[0] - parent_radius
        parent_x_max = parent_central_position[0] + parent_radius
        parent_y_min = parent_central_position[1] - parent_radius
        parent_y_max = parent_central_position[1] + parent_radius
        
        parent_width = parent_x_max - parent_x_min
        parent_height = parent_y_max - parent_y_min

        frame_extent = self.frame.extent
        frame_width = frame_extent[1][0] - frame_extent[0][0]
        frame_height = frame_extent[1][1] - frame_extent[0][1]

        scaling_factor_x = parent_width / frame_width
        scaling_factor_y = parent_height / frame_height

        self.frame.set_scale_factor(scaling_factor_x, scaling_factor_y)
        
        for id, community in self.communities_dict.items():
            community.set_scale_factor(scaling_factor_x, scaling_factor_y)
    

    def balaban_edge_crossings(self, positions, weighted = False):
        edges_intersections = self.get_community_graph().get_edge_edge_intersections(positions)
        community_intersections = self.get_community_graph().get_edge_community_intersection(self.communities_dict, positions)

        # Count intersections
        count_edges_intersections = len(edges_intersections)
        count_community_intersections = len(community_intersections)

        # Calculate normalization factors
        num_edges = self.get_community_graph().g_c.number_of_edges()
        num_nodes = self.get_community_graph().g_c.number_of_nodes()

        # Normalization factor: Total possible intersections
        max_possible_edge_intersections = num_edges * (num_edges - 1) / 2
        max_possible_community_intersections = num_edges * num_nodes

        # Normalize counts (avoid division by zero)
        normalized_edges_intersections = count_edges_intersections / max_possible_edge_intersections if max_possible_edge_intersections > 0 else 0
        normalized_community_intersections = count_community_intersections / max_possible_community_intersections if max_possible_community_intersections > 0 else 0

        # Optionally, calculate weighted intersections (if weighted=True)
        if weighted:
            weighted_edges_intersections = sum(intersection[2] + intersection[3] for intersection in edges_intersections) / (2 * max_possible_edge_intersections) if max_possible_edge_intersections > 0 else 0
            weighted_community_intersections = sum(intersection[2] for intersection in community_intersections) / max_possible_community_intersections if max_possible_community_intersections > 0 else 0
            return normalized_edges_intersections, normalized_community_intersections, weighted_edges_intersections, weighted_community_intersections

        return (normalized_edges_intersections + normalized_community_intersections)/2
    
    def update_community_positions(self, key, comm, rescaled):
        pos = None
        if rescaled:
            pos = self.frame.get_rescaled_pos(self.community_positions[key], 
                                                        self.upper_community.get_central_position(), 
                                                        self.upper_community.radius)
        else:
            pos = self.community_positions[key]
        
        comm.set_central_position(pos)


    def set_node_positions(self, calculate_node_positions, rescaled):
        node_positions = {}
        for key, comm in self.communities_dict.items():
            self.update_community_positions(key, comm, rescaled)
            if calculate_node_positions:
                p =  comm.get_comm_node_positions(self.graph)
                node_positions.update(p)
        
        return node_positions