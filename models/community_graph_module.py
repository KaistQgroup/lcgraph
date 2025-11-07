import networkx as nx
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

class CommunityGraph():

    def __init__(self, graph, communities_dict):
        self.get_community_graph(graph, communities_dict)

    def get_community_graph(self, graph, communities_dict):
        self.g_c = nx.Graph()  # Create a new graph where communities are nodes

        # Each community is represented as a node
        for community_id, community in communities_dict.items():
            self.g_c.add_node(community_id)

        # Add weighted edges between communities if their nodes are connected in the original graph
        for community_id1, community1 in communities_dict.items():
            for community_id2, community2 in communities_dict.items():
                if community_id1 != community_id2:
                    weight = 0
                    for node1 in community1.nodes:
                        for node2 in community2.nodes:
                            if graph.has_edge(node1, node2):
                                weight += 1  # Increment weight if there's an edge between the two nodes
                    if weight > 0:
                        if self.g_c.has_edge(community_id1, community_id2):
                            self.g_c[community_id1][community_id2]['weight'] += weight
                        else:
                            self.g_c.add_edge(community_id1, community_id2, weight=weight)

        

    def get_disconnected_components(self):
        connected_components = list(nx.connected_components(self.g_c))
        return [list(c)[0] for c in connected_components if len(c) == 1]
    

    def get_edge_edge_intersections(self, positions):
        # Convert edges to LineString geometries and create a spatial index
        line_segments = [
            (edge, LineString([positions[edge[0]], positions[edge[1]]]))
            for edge in self.g_c.edges()
            if self.is_valid_geometry(LineString([positions[edge[0]], positions[edge[1]]]))
        ]
        
        lines = [line for _, line in line_segments]
        line_to_edge = {line: edge for edge, line in line_segments}
        tree = STRtree(lines)

        intersections = []
        for i, line in enumerate(lines):
            edge1 = line_to_edge[line]
            weight1 = self.g_c.get_edge_data(*edge1)['weight']  # Get weight of the first edge

            for other in tree.query(line):
                if line != lines[other] and line.intersects(lines[other]):
                    intersection = line.intersection(lines[other])
                    if not intersection.is_empty:
                        edge2 = line_to_edge[lines[other]]
                        weight2 = self.g_c.get_edge_data(*edge2)['weight']  # Get weight of the second edge
                        intersections.append((line, lines[other], weight1, weight2, intersection))

        return intersections

    def get_edge_community_intersection(self, communities_dict, positions):
        # Prepare circles around each community
        community_circles = {
            community_id: Point(pos).buffer(community.radius)
            for community_id, pos in positions.items()
            if (community := communities_dict.get(community_id))
        }

        # Convert edges to LineString geometries
        intersections = []
        for edge in self.g_c.edges():
            line = LineString([positions[edge[0]], positions[edge[1]]])
            weight = self.g_c.get_edge_data(*edge)['weight']
            for community_id, circle in community_circles.items():
                if edge[0] != community_id and edge[1] != community_id and line.intersects(circle):
                    intersection = line.intersection(circle)
                    if not intersection.is_empty:
                        intersections.append((line, community_id, weight, intersection))
        return intersections
    
    def is_valid_geometry(self, geometry):
        if not geometry.is_valid:
            print(f"Invalid geometry")
            return False
        return True

