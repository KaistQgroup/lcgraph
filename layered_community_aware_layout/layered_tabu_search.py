import numpy as np
from collections import deque
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree
from utils.utils import safe_str
from concurrent.futures import ThreadPoolExecutor, as_completed
import parmap
import time

class LayeredTabuSearch:
    def __init__(self, network):
        self.network = network
        self.num_iterations = 150
        self.all_neighbors = []
    
    def optimize_parmap(self, comm_sets_list):
        results = parmap.map(self.optimize_community_, comm_sets_list, pm_pbar=True, pm_processes=16)
        
        # Update community positions in the main process
        for uid, best_solution in results:
            for item in comm_sets_list:
                if item.id == uid:
                    item.set_community_positions(best_solution, add_disconnected = True)
                    break

    def optimize_community_(self, item):
        uid, positions = item.id, item.get_community_positions()

        if len(positions) == 1:
            return uid, positions
                
        best_solution = positions
        best_cost = self.objective_function(item, positions)
        tabu_list = deque(maxlen=100)

        for _i in range(self.num_iterations):
            neighborhood = self.get_neighborhood(positions)
            neighborhood = [sol for sol in neighborhood if sol not in tabu_list]

            if not neighborhood:
                break

            new_solution = min(neighborhood, key=lambda pos: self.objective_function(item, pos))
            new_cost = self.objective_function(item, new_solution)

            if new_cost < best_cost:
                best_solution = new_solution
                best_cost = new_cost

            tabu_list.append(new_solution)
            positions = new_solution

        return uid, best_solution
        
    
    def objective_function(self, item, positions):
        return item.balaban_edge_crossings(positions)
    
    
    def get_neighborhood(self, positions):
        keys = list(positions.keys())
        neighborhood = []

        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                new_solution = positions.copy()
                new_solution[keys[i]], new_solution[keys[j]] = new_solution[keys[j]], new_solution[keys[i]]
                if new_solution not in self.all_neighbors:
                    neighborhood.append(new_solution)
                    self.all_neighbors.append(new_solution)

        return neighborhood
    

