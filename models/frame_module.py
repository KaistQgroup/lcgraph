import numpy as np
from scipy.spatial import KDTree
import networkx as nx
import numpy as np
from scipy.spatial.distance import cdist

class Frame():

    def __init__(self, communities, level = None, hexa_grid = False, frame_extent = None, spring_grids = False, graph = None):
        # if frame_extent is None:
        #     frame_extent = [[-4000, 4000], [-2800, 2800]]

        self.disconnected_grids_array = None

        if hexa_grid:
            self.max_radius = int(np.ceil(max([communities[c].radius for c in communities]))) + 800
            num_points = len(communities)
            self.grids = self.create_hexagonal_grid(num_points)
        elif spring_grids:
            self.max_radius = int(np.ceil(max([communities[c].radius for c in communities])))
            self.grids = self.create_spring_grids(graph, self.max_radius)
        else:

            self.min_radius = int(np.ceil(min([communities[c].radius for c in communities])))

            if frame_extent == None:
                if level == 1:
                    sqauer_area = [np.power(((communities[c].radius + 1 * (self.min_radius))) * 2, 2) for c in communities]
                    sqauer_area_sum = sum(sqauer_area)
                    diameter = int(np.sqrt(sqauer_area_sum))
                    print(diameter)
                    frame_extent = [[-diameter, diameter], [-diameter, diameter]]
                else:
                    frame_extent = [[-4000, 4000], [-2800, 2800]]
            
            self.extent = frame_extent
            self.width = self.extent[0][1] - self.extent[0][0]
            self.height = self.extent[1][1] - self.extent[1][0]

            
            self.grids = self.create_grids(self.min_radius)
            self.spiral_order = self.get_spiral_order()


    def create_grids(self, min_radius):
        """Create a grid of points within the screen size."""                   

        x_points = np.arange(self.extent[0][0] + min_radius, self.extent[0][1] - min_radius, min_radius)
        y_points = np.arange(self.extent[1][0] + min_radius, self.extent[1][1] - min_radius, min_radius)
        self.grids_array = np.array([(x, y) for x in x_points for y in y_points])

        grids_2d = self.grids_array.reshape((len(x_points), len(y_points), 2))

        return grids_2d
    
    def create_hexagonal_grid(self, num_points):

        x_points, y_points = [], []
        if num_points == 1:
            x_points, y_points = [self.max_radius / 2], [self.max_radius * np.sqrt(3) / 4]

        if num_points == 11:
            # x_points, y_points = self.generate_polygon_points()
            x_points = [3 * self.max_radius, 5 * self.max_radius, 2 * self.max_radius, 4 * self.max_radius, 6 * self.max_radius, 2 * self.max_radius, 4 * self.max_radius, 6 * self.max_radius, 3 * self.max_radius, 5 * self.max_radius, 4 * self.max_radius]
            y_points = [0, 0, 1 * self.max_radius, 1 * self.max_radius, 1 * self.max_radius, 2 * self.max_radius, 2 * self.max_radius, 2 * self.max_radius, 3 * self.max_radius, 3 * self.max_radius, 4 * self.max_radius ]


        else:
            self.rows = int(np.ceil(np.sqrt(num_points / np.sqrt(3) * 2)))
            for row in range(self.rows):
                cols = int(np.ceil(num_points / self.rows))
                
                for col in range(cols):
                    x_points.append(self.max_radius * (col + (row % 2) * 0.5))
                    y_points.append(self.max_radius * row * np.sqrt(3) / 2)
                    
                    if len(x_points) >= num_points:
                        break
        
            last_row_points = len(x_points) % cols
            if last_row_points == 1:
                x_points[-1] = max(x_points) / 2  # Center the last point in its row

        self.bottom_left = (min(x_points), min(y_points))
        self.top_right = (max(x_points), max(y_points))

        self.grids_array = list(zip(x_points, y_points))

        self.extent = [[min(x_points) - self.max_radius, min(y_points) - self.max_radius ], [max(x_points) + self.max_radius, max(y_points) + self.max_radius]]
                    
        return self.grids_array
    
    
    def get_spiral_order(self):
        """Get the spiral order of grid points starting from the center."""
        rows, cols, _ = self.grids.shape
        center = (rows // 2, cols // 2)
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        direction_idx = 0

        visited = np.zeros((rows, cols), dtype=bool)
        spiral_order = []

        r, c = [0, 0]
        for _ in range(rows * cols):
            spiral_order.append(self.grids[r, c])
            visited[r, c] = True

            next_r = r + directions[direction_idx][0]
            next_c = c + directions[direction_idx][1]

            if (0 <= next_r < rows and 0 <= next_c < cols and not visited[next_r, next_c]):
                r, c = next_r, next_c
            else:
                direction_idx = (direction_idx + 1) % 4
                r += directions[direction_idx][0]
                c += directions[direction_idx][1]

        return spiral_order
    
    def select_evenly_spread_grids(self, n):
        """Select n evenly spread grid points from self.grids_array."""
        if self.grids_array is None:
            raise ValueError("Grids array has not been initialized.")

        total_cells_x = self.grids.shape[0]
        total_cells_y = self.grids.shape[1]
        
        # Calculate the step size for both dimensions
        step_size_x = max(1, total_cells_x // int(np.ceil(np.sqrt(n))))
        step_size_y = max(1, total_cells_y // int(np.ceil(np.sqrt(n))))

        selected_grids = []
        for i in range(0, total_cells_x, step_size_x):
            for j in range(0, total_cells_y, step_size_y):
                selected_grids.append(self.grids_array[i, j])
                if len(selected_grids) == n:
                    break
            if len(selected_grids) == n:
                break
        
        return np.array(selected_grids)
    

    def set_scale_factor(self, scaling_factor_x, scaling_factor_y):
        self.scaling_factor_x = scaling_factor_x
        self.scaling_factor_y = scaling_factor_y

    def get_rescaled_pos(self, position, parent_pos, parent_radius):
        parent_x_min = parent_pos[0] - parent_radius
        parent_y_min = parent_pos[1] - parent_radius
        return (parent_x_min + (position[0] - self.extent[0][0]) * self.scaling_factor_x, 
                      parent_y_min + (position[1] - self.extent[0][1]) * self.scaling_factor_y) 
    
    def extract_sub_grids(self, centralPosition, radius):
        """Extract a sub-grid based on the extent."""
            # Find the nearest grid point to the target_point
        tree = KDTree(self.grids_array)
        distance, index = tree.query(centralPosition)
        nearest_point = self.grids_array[index]
        
        # Find the 2D index
        idx = np.argwhere(np.all(self.grids == nearest_point, axis=2))[0]
        d_trunc = int(np.trunc(radius / self.min_radius)) - 2

        gshape = self.grids.shape

        sub_grids = self.grids[max(idx[0] - d_trunc, 0):min(idx[0] + d_trunc + 1, gshape[0]), max(idx[1] - d_trunc, 0):min(idx[1] + d_trunc + 1, gshape[1]), :]
        
        return sub_grids
    
    def extract_layer_disconnected_grids(self, disconnected_count):
                
        bottom_left = self.bottom_left
        top_right = self.top_right
        
        x_bottom_left, y_bottom_left = bottom_left
        x_top_right, y_top_right = top_right

        # x coordinate for the disconnected grids
        y_disconnected = y_bottom_left - (self.max_radius)
        
        # Generate `disconnected_count` y coordinates between `y_bottom_left` and `y_top_right`
        x_disconnected = np.linspace(x_bottom_left, x_top_right, disconnected_count)
        
        # Create the list of disconnected grid positions
        disconnected_positions = [(x, y_disconnected) for x in x_disconnected]
        
        self.disconnected_grids_array = disconnected_positions
    
    def generate_polygon_points(self, num_sides = 9 , side_length = 2200):
        """
        Generate the vertices of a regular polygon and additional points.

        Parameters:
        num_sides (int): Number of sides of the polygon.
        side_length (float): Length of each side of the polygon.

        Returns:
        tuple: Two lists containing the x and y coordinates of the points.
        """
        angle = 2 * np.pi / num_sides
        vertices = [(np.cos(i * angle), np.sin(i * angle)) for i in range(num_sides)]
        vertices = np.array(vertices) * side_length

        center_x = np.mean(vertices[:, 0])
        center_y = np.mean(vertices[:, 1])
        half_side_length = side_length / 3

        new_points = [
            (center_x + 100, center_y + half_side_length),
            (center_x - 100, center_y - half_side_length)
        ]

        all_points = np.append(vertices, new_points, axis=0)
        x_coords = [point[0] for point in all_points]
        y_coords = [point[1] for point in all_points]

        return x_coords, y_coords
    

    def get_grids_array(self):
        return self.grids_array.copy()
    
    def get_disconnected_grids_array(self):
        if self.disconnected_grids_array is not None:
            return self.disconnected_grids_array.copy()
        else:
            return self.disconnected_grids_array
    

    def create_spring_grids(self, G, desired_min_distance):
        def find_min_distance(pos):
            """Finds the minimum distance between all pairs of nodes using cdist."""
            pos_array = np.array(list(pos.values()))
            distance_matrix = cdist(pos_array, pos_array, metric='euclidean')
            np.fill_diagonal(distance_matrix, np.inf)
            min_distance = distance_matrix.min()
            return min_distance

        pos = nx.spring_layout(G)  
        min_distance = find_min_distance(pos)  
        scale_factor = desired_min_distance / min_distance  
        pos_array = np.array(list(pos.values()))
        pos_rescaled = nx.rescale_layout(pos_array, scale=scale_factor)  

        x_points = pos_rescaled[:, 0]
        y_points = pos_rescaled[:, 1]
        
        self.bottom_left = (min(x_points), min(y_points))
        self.top_right = (max(x_points), max(y_points))

        self.grids_array = list(zip(x_points, y_points))

        self.extent = [
            [min(x_points) - self.max_radius, min(y_points) - self.max_radius],
            [max(x_points) + self.max_radius, max(y_points) + self.max_radius]
        ]

        return self.grids_array



            


