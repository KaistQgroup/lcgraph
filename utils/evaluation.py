import numpy as np
from scipy.spatial import distance_matrix, ConvexHull
import networkx as nx
from shapely.geometry import LineString
from shapely.strtree import STRtree
import itertools
import parmap
from collections import defaultdict
from shapely.geometry import Polygon, MultiPolygon, Point



class Metrics():
    def __init__(self, network, positions, level = False, layered = False, original_positions=None):
        self.network = network
        self.graph = network.graph
        self.positions = positions
        if layered:
            self.communities = network.all_communities
        else:
            self.communities = network.communities

        if level:
            self.levels = network.levels
        
        self.layered = layered
        self.original_positions=original_positions

    def node_spread(self, community_dict):
        if self.layered:
            # Dictionary to store distances for each community
            distances = {com.uId: [] for com in self.communities if com.current_level == self.network.max_level}
            # Precompute positions of community centers
            community_centers = {com.uId: self.positions[com.getCentralPoint()] for com in self.communities}

        else:
            # Dictionary to store distances for each community
            distances = {com_id: [] for com_id in self.communities}
            # Precompute positions of community centers
            community_centers = {com_id: self.positions[community.getCentralPoint()] for com_id, community in self.communities.items()}
        
        
        # Iterate through all nodes in the graph
        for node in self.graph.nodes():
            com_id = community_dict[node]
            center_pos = community_centers[com_id]
            node_pos = self.positions[node]
            
            # Calculate the distance between the node and the community center
            node_distance = np.linalg.norm(np.array(node_pos) - np.array(center_pos))
            distances[com_id].append(node_distance)
        
        # Calculate the average distance for each community
        com_distances = [np.mean(distances[com_id]) for com_id in distances]
        
        # Calculate the overall average node spread
        overall_node_spread = np.mean(com_distances)
        
        return overall_node_spread
    
    def node_occlusions(self, thr=1.0):
        
        # Extract positions of nodes into a numpy array
        nodes = np.array([self.positions[node] for node in self.graph.nodes()])

        # Calculate the distance matrix
        dm = distance_matrix(nodes, nodes)

        # Ignore self-pairs by setting the diagonal to infinity
        np.fill_diagonal(dm, np.inf)

        # Count the number of occlusions (pairs with distance below the threshold)
        occlusions = np.sum(dm < thr) / 2  # Each pair is counted twice

        # Calculate the total number of node pairs
        total_pairs = len(nodes) * (len(nodes) - 1) / 2

        # Return the node occlusion ratio
        return occlusions / total_pairs
    
    def spatial_autocorrelation(self, community_dict, radius=0.1):
        def normalized_distance(pos1, pos2):
            return np.linalg.norm(np.array(pos1) - np.array(pos2)) / radius
        
        def is_same_community(node1, node2):
            return community_dict[node1] == community_dict[node2]
        
        if self.original_positions:
            position_array = self.original_positions
        else:
            position_array = self.positions

        nodes = list(self.graph.nodes())
        positions = np.array([position_array[node] for node in nodes])
        dm = distance_matrix(positions, positions)

        autocorrelations = []
        for i, node in enumerate(nodes):
            neighbors = np.where(dm[i] < radius)[0]
            if len(neighbors) == 0:
                continue

            norm_dists = dm[i][neighbors] / radius
            sum_weights = 0
            sum_autocorr = 0
            for idx, neighbor_idx in enumerate(neighbors):
                if i == neighbor_idx:
                    continue
                weight = 1 - norm_dists[idx]
                sum_weights += weight
                sum_autocorr += weight * (1 if not is_same_community(node, nodes[neighbor_idx]) else 0)
            
            if sum_weights > 0:
                Ci = sum_autocorr / sum_weights
                autocorrelations.append(Ci)

        return np.mean(autocorrelations) if autocorrelations else 0
    

    def community_entropy(self, community_dict, grid_size=(30, 30)):

        if self.original_positions:
            position_array = self.original_positions
        else:
            position_array = self.positions

        # Determine the bounds of the grid
        x_coords = [pos[0] for pos in position_array.values()]
        y_coords = [pos[1] for pos in position_array.values()]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)

        # Determine the size of each cell in the grid
        x_step = (max_x - min_x) / grid_size[0]
        y_step = (max_y - min_y) / grid_size[1]

        # Initialize a dictionary to store counts of nodes per region and community
        region_counts = defaultdict(lambda: defaultdict(int))

        # Assign each node to a region
        for node, pos in position_array.items():
            x_idx = int((pos[0] - min_x) / x_step)
            y_idx = int((pos[1] - min_y) / y_step)
            region = (x_idx, y_idx)
            community_id = community_dict[node]
            region_counts[region][community_id] += 1

        # Calculate the entropy for each region
        region_entropies = []
        for region, counts in region_counts.items():
            total_nodes = sum(counts.values())
            proportions = [count / total_nodes for count in counts.values()]
            entropy = -sum(p * np.log2(p) for p in proportions if p > 0)
            region_entropies.append(entropy)

        # Average the entropies across all regions
        overall_entropy = np.mean(region_entropies) if region_entropies else 0

        return overall_entropy
    
    def group_overlap(self, community_dict):
        # Collect points for each community
        community_points = defaultdict(list)
        for node, community in community_dict.items():
            community_points[community].append(self.positions[node])
        
        # Calculate convex hulls for each community
        convex_hulls = {}
        for community, points in community_points.items():
            if len(points) >= 3:
                hull = ConvexHull(points)
                vertices = hull.vertices
                np.append(vertices, vertices[0])
                convex_hulls[community] = Polygon([points[vertex] for vertex in vertices])
        
        # Calculate the overlap for each community
        # Calculate the overlap for each community
        overlaps = []
        total_area = 0
        intersection_area = 0
        hull_list = list(convex_hulls.values())
        
        for i, hull_i in enumerate(hull_list):
            total_area += hull_i.area
            for j, hull_j in enumerate(hull_list):
                if i != j:
                    if hull_i.intersects(hull_j):
                        intersection = hull_i.intersection(hull_j)
                        intersection_area += intersection.area

        # Calculate the Group Overlap
        if total_area > 0:
            group_overlap = (intersection_area / total_area)
        else:
            group_overlap = 0

        return group_overlap
    

    def balaban_edge_crossings(self):
        # Extract edges and self.positions
        edges = list(self.graph.edges())
        edge_positions = {node: (self.positions[node][0], self.positions[node][1]) for node in self.graph.nodes()}

        # Convert edges to line segments
        line_segments = []
        for edge in edges:
            start, end = edge
            line_segments.append((edge_positions[start], edge_positions[end]))

        # Convert line segments to Shapely LineString objects
        lines = [LineString(segment) for segment in line_segments]
        print(len(lines))

        # Create an STRtree for efficient spatial querying
        tree = STRtree(lines)

        num_chunks = 12  # Adjust this number based on your available CPU cores
        chunks = [list(enumerate(lines))[i::num_chunks] for i in range(num_chunks)]

        # Use parmap to apply the find_intersections function to each chunk in parallel
        results = parmap.map(self.find_intersections, chunks, lines, tree, pm_pbar=True, pm_processes=num_chunks)

        # Combine the results from all chunks
        intersections = set()
        for result in results:
            intersections.update(result)

        total_pairs = len(edges) * (len(edges) - 1) / 2
        return len(intersections) / total_pairs
    
    def find_intersections(self, chunk, lines, tree):
        intersections = set()
        for i, line in chunk:
            for other in tree.query(line):
                if i < other and line.intersects(lines[other]):
                    intersection = line.intersection(lines[other])
                    if not intersection.is_empty:
                        intersections.add(intersection)
        return intersections

