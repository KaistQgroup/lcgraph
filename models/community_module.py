import numpy as np
from utils.utils import safe_str
import networkx as nx

class Community():    
    def __init__(self, id, nodes, current_level, upper_community=None, upper_level=None, sub_communities=None):
        self.id = id
        self.nodes = nodes
        self.nodes_len = len(nodes)
        self.area, self.radius = self.compute_area_radius() 
        self.current_level = current_level
        self.upper_community = upper_community
        self.upper_level = upper_level
        self.sub_communities = sub_communities
        self.centralPosition = [0, 0]
        self.uId = safe_str(current_level) + "_" + safe_str(upper_community) + "_" + safe_str(id)
        self.scaling_factor_x = 1
        self.scaling_factor_y = 1
        self.node_positions = None
    
    def compute_area_radius(self):
        k = 7000
        base_factor = 100
        basic_distance = 10

        area = k * (self.nodes_len + 1) ** 0.8
        radius = base_factor + np.sqrt(area / np.pi)

        return area, radius
    
    def set_central_position(self, position):
        self.centralPosition = position
    
    def get_central_position(self):
        return self.centralPosition
    
    def setGridExtent(self, extent):
        self.gridExtent = extent
    
    def getGridExtent(self):
        return self.gridExtent
    
    def get_central_node(self, degree_centralities):
        return max(self.nodes, key=lambda node: degree_centralities[node])
    
    def set_scale_factor(self, scaling_factor_x, scaling_factor_y):
        self.scaling_factor_x = scaling_factor_x
        self.scaling_factor_y = scaling_factor_y
    
    def get_radius(self):
        radius = self.radius * min(self.scaling_factor_x, self.scaling_factor_y)
        return radius
    
    def set_is_disconnected(self, is_disconnected):
        self.is_disconnected = is_disconnected
    
    def get_is_disconnected(self):
        return self.is_disconnected
    
    def re_scale_sub_communities(self, maxLevel, levels, all_frames):
        if self.upper_community != None and self.upper_level != None:
            
            parent_level = levels[self.current_level - 2]
            
            central_position = self.get_central_position()
            parent_key = f"{self.upper_level}_{self.upper_level - 1}_{self.upper_community}"
            parent_community = parent_level['communities'][parent_key]
            parent_central_position = parent_community.get_central_position()
            parent_radius = parent_community.radius

            parent_x_min = parent_central_position[0] - parent_radius
            parent_x_max = parent_central_position[0] + parent_radius
            parent_y_min = parent_central_position[1] - parent_radius
            parent_y_max = parent_central_position[1] + parent_radius
            
            parent_width = parent_x_max - parent_x_min
            parent_height = parent_y_max - parent_y_min

            frame_extent = all_frames[parent_key].extent
            frame_width = frame_extent[1][0] - frame_extent[0][0]
            frame_height = frame_extent[1][1] - frame_extent[0][1]

            scaling_factor_x = parent_width / frame_width
            scaling_factor_y = parent_height / frame_height

            new_x = parent_x_min + (central_position[0] - frame_extent[0][0]) * scaling_factor_x
            new_y = parent_y_min + (central_position[1] - frame_extent[0][1]) * scaling_factor_y

            self.set_central_position([new_x, new_y])
            self.radius *= min(scaling_factor_x, scaling_factor_y)
    
    
    def get_comm_node_positions(self, graph, central_position = None):
        if self.node_positions == None:
            subgraph = graph.subgraph(self.nodes)  # Create a subgraph for the current community
            if central_position:  # Get the position of the community center and its area
                center = central_position
            else:
                center = self.get_central_position()    
            # if center[0] == 0 and center[1] == 0:
            #     return {}
            radius = self.get_radius() - 1
            self.node_positions = nx.spring_layout(subgraph, center=center, scale=radius, seed=123)   # Use a force-directed layout within the community circle
            
        return self.node_positions