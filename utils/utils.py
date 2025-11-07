
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, LineString
import random
import matplotlib.colors as mcolors
import json


laws_dict = {
    "1_0_0": "Criminal Law",
    "1_0_8": "National Public Officials Act",
    "1_0_15": "Family-Friendly Act",
    "1_0_3": "Judicial Police Act",
    "1_0_5": "Labor Standards Act",
    "1_0_2": "Autonomous Fisheries Act",
    "1_0_12": "State Property Act",
    "1_0_4": "Construction Laws Act",
    "1_0_7": "Commercial Act",
    "1_0_1": "Special Taxation Act",
    "1_0_9": "Property Disposition Act",
    "1_0_6": "Japanese Property Act",
    "1_0_10": "Medical Defection Act",
    "1_0_11": "Special Marriage Act",
    "1_0_13": "Independence Bonds Act",
    "1_0_14": "Geochang Honor Act",
    "1_0_16": "No Gun Ri Act",
    "1_0_17": "Financial Stability Act",
    "1_0_18": "Defense Reform Act",
    "1_0_19": "Astronomy Act",
    "1_0_20": "HPC Promotion Act",
    "1_0_21": "Sakhalin Support Act",
    "1_0_22": "Workers in Germany Act",
    "1_0_23": "PFC Promotion Act",
    "2_0_0": "Logistics Policy Act",
    "2_0_1": "Criminal Law",
    "2_0_2": "Industrial Standardization Act",
    "2_0_3": "Industrial Safety Act",
    "2_0_4": "Public Information Act",
    "2_0_5": "Higher Education Act",
    "2_0_6": "Small and Medium Enterprises Act",
    "2_0_7": "Public Institutions Act",
    "2_0_8": "Defense Business Act",
    "2_0_9": "Engineering Industry Promotion Act",
    "2_0_10": "National Finance Act",
    "2_0_11": "Land and Transport Science Act",
    "2_0_12": "Fire Service Act",
    "2_8_0": "Elementary and Secondary Education Act",
    "2_8_1": "Private School Act",
    "2_8_2": "National Public Officials Act",
    "2_8_3": "Lifelong Education Act",
    "2_8_4": "Police Uniform and Equipment Regulation Act",
    "2_8_5": "Road Traffic Act",
    "2_8_6": "Veterans Support Act",
    "2_8_7": "Youth Protection Act",
    "2_8_8": "Low Birthrate and Aging Society Act",
    "2_8_9": "Rural Healthcare Act",
    "2_8_10": "Disqualification Clauses Act",
    "2_15_0": "Family-Friendly Act",
    "2_3_0": "Disaster Management Act",
    "2_3_1": "Korean Medicine Promotion Act",
    "2_3_2": "Information Network Act",
    "2_3_3": "High-Rise and Underground Complex Disaster Management Act",
    "2_3_4": "Zoo and Aquarium Management Act",
    "2_3_5": "E-Government Act",
    "2_3_6": "Hansen's Disease Victims Support Act",
    "2_3_7": "Public Interest Corporations Act",
    "2_3_8": "National Sports Promotion Act",
    "2_3_9": "Judicial Police Act",
    "2_3_10": "Cultural Diversity Act",
    "2_3_11": "Fair Hiring Act",
    "2_3_12": "Misdemeanor Punishment Act",
    "2_3_13": "Volunteer Activities Act",
    "2_3_14": "Food Sanitation Act",
    "2_3_15": "3D Printing Industry Promotion Act",
    "2_3_16": "Resident Training Act",
    "2_5_0": "Labor Standards Act",
    "2_5_1": "Employment Security Act",
    "2_5_2": "Work-Learning Dual System Act",
    "2_5_3": "Lifelong Vocational Competency Act",
    "2_5_4": "Construction Workers Employment Act",
    "2_5_5": "Labor Union Act",
    "2_5_6": "Employment Insurance Act",
    "2_5_7": "Kaesong Industrial Zone Support Act",
    "2_2_0": "Autonomous Fisheries Act",
    "2_12_0": "State Property Act",
    "2_12_1": "Jeju Special Act",
    "2_12_2": "Sewerage Act",
    "2_12_3": "Asian Culture City Act",
    "2_12_4": "Public Property Act",
    "2_12_5": "Road Act",
    "2_12_6": "Land Acquisition and Compensation Act",
    "2_12_7": "Land Planning and Utilization Act",
    "2_12_8": "Real Estate Price Disclosure Act",
    "2_4_0": "Local Cultural Centers Promotion Act",
    "2_4_1": "Foreign Missions Property Act",
    "2_4_2": "Toll Road Act",
    "2_4_3": "Forest Protection Act",
    "2_4_4": "Administrative Procedures Act",
    "2_4_5": "Construction Laws Act",
    "2_4_6": "Budget Accounting Act",
    "2_4_7": "Mining Safety Act",
    "2_4_8": "Goods Information Management Act",
    "2_4_9": "Licensing Reform Act",
    "2_4_10": "Korea Tourism Organization Act",
    "2_4_11": "Assembly and Demonstration Act",
    "2_4_12": "Breakwater Management Act",
    "2_4_13": "Public Health Scholarship Act",
    "2_4_14": "Specific Buildings Act",
    "2_7_0": "National Tax Collection Act",
    "2_7_1": "Insurance Act",
    "2_7_2": "Capital Markets Act",
    "2_7_3": "National Health Insurance Act",
    "2_7_4": "Civil Procedure Act",
    "2_7_5": "Temporary Special Act on National and Public Property",
    "2_7_6": "Fair Subcontracting Act",
    "2_7_7": "Commercial Act",
    "2_1_0": "Income Tax Act",
    "2_1_1": "Customs Act",
    "2_1_2": "Agricultural and Food Industry Act",
    "2_1_3": "Apartment Management Act",
    "2_1_4": "Public Holidays Act",
    "2_1_5": "Foreign Investment Promotion Act",
    "2_1_6": "Mining Damage Prevention Act",
    "2_1_7": "Special Taxation Act",
    "2_1_8": "Local Autonomy Act",
    "2_1_9": "National Bond Act",
    "2_1_10": "Goods Tax Act",
    "2_1_11": "Postal Service Act",
    "2_9_0": "Property Disposition Act",
    "2_9_1": "Farmland Ownership Act",
    "2_9_2": "Real Estate Ownership Act",
    "2_6_0": "Japanese Property Act",
    "2_10_0": "Medical Defection Act",
    "2_11_0": "Special Marriage Act",
    "2_13_0": "Independence Bonds Act",
    "2_14_0": "Geochang Honor Act",
    "2_16_0": "No Gun Ri Act",
    "2_17_0": "Financial Stability Act",
    "2_18_0": "Defense Reform Act",
    "2_19_0": "Astronomy Act",
    "2_20_0": "HPC Promotion Act",
    "2_21_0": "Sakhalin Support Act",
    "2_22_0": "Workers in Germany Act",
    "2_23_0": "PFC Promotion Act"
    }


color_codes = [
    "#BD75BD",  # 0
    "#B01743",  # 1Dark Red
    "#FD4659",  # 2Watermelon
    "#581845",  # 3Dark Purple
    "#8E44AD",  # 4Purple
    "#8D8B55",  # 5Amethyst
    "#226699",  # 6Blue
    "#3498DB",  # 7Light Blue
    "#1F618D",  # 8Dark Blue
    "#16A085",  # 9Teal
    "#1ABC9C",  # 10Light Teal
    "#27AE60",  # 11Green
    "#2ECC71",  # 12Light Green
    "#de7518",  # 13Orange
    "#B53b2d",  # 14Firebrick
    "#F39C12",  # 15Dark Red
    "#b5a751",  # 16Golden Rod
    "#6C6C6C",  # 17Gray
    "#D4AC0D",  # 18olden Yellow
    "#873600",  # 19Dark Teal
    "#273746",  # 20Dark Slate
    "#C4A484",  # 21Dark Violet
    "#D2527F",  # 22Rose
    "#DE7E5D",  # 23Deep Blue
    "#873600",   # 24Brown
    "#873600",   # 24Brown
    "#b5a751",  # 16Golden Rod
    "#6C6C6C",  # 17Gray
    "#D4AC0D",  # 18olden Yellow
    "#873600",  # 19Dark Teal
    "#273746",  # 20Dark Slate
    "#C4A484",  # 21Dark Violet
    "#D2527F",  # 22Rose
    "#DE7E5D",  # 23Deep Blue
    "#873600",   # 24Brown
    "#873600",   # 24Brown
    "#873600",   # 24Brown
    "#873600",   # 24Brown
    "#b5a751",  # 16Golden Rod
    "#6C6C6C",  # 17Gray
    "#D4AC0D",  # 18olden Yellow
    "#873600",  # 19Dark Teal
    "#273746",  # 20Dark Slate
    "#C4A484",  # 21Dark Violet
    "#D2527F",  # 22Rose
    "#DE7E5D",  # 23Deep Blue
    "#873600",   # 24Brown
    "#873600"   # 24Brown
]

def community_output_json_generation(network, ldb, fileName):
        # Initialize containers
    community_nodes = []
    community_edges = []
    nodes = []
    edges = []
    colors = None


    colors = color_codes #generate_random_color(len(network.levels[1][0].get_communities_dict())) #color_codes
    outer_extent = network.levels[1][0].frame.extent

    # Process each level in `levels`
    for level in range(1, network.max_level + 1):
        community_sets = network.levels[level]
        is_last_layer = False
        is_first_layer = False
        for community_set in community_sets:
            community_dict = community_set.get_communities_dict()
            community_graph = community_set.get_community_graph().g_c
            color = colors[int(community_set.id.split("_")[2])]

            if level == 1:
                is_first_layer = True

            if level == network.max_level:
                is_last_layer = True

            comunity_node_json, node_json = get_community_nodes_json(community_dict, is_last_layer, is_first_layer, ldb.node_titles, outer_extent)
            community_nodes.extend(comunity_node_json)
            nodes.extend(node_json)
            
            community_edges.extend(get_community_edges_json(community_graph, community_dict))

    edges = get_graph_edges_json(ldb.unique_edges, ldb.citation_type)

    return get_output(community_nodes, community_edges, nodes, edges, fileName, isLayered=True)


def getArticleOutput(graph, fileName):
    pass

def get_community_nodes_json(communities_dict, is_last_layer, is_first_layer, node_titles, extent):
    communities_nodes = []
    nodes = []
    for id, com in communities_dict.items():
        if is_first_layer:
            color = color_codes[int(id.split("_")[2])]
        else:
            color = color_codes[int(id.split("_")[1])]
        
        n_pos = get_normalize_pos(com.get_central_position(), extent, com.radius)
        communities_nodes.append({
            'id': id,
            'x': float(com.get_central_position()[0]),
            'y': float(com.get_central_position()[1]),
            'n_x': float(n_pos[0]),
            'n_y': float(n_pos[1]),
            'community': id,
            'radius': com.radius,
            'n_radius': n_pos[2],
            'color': color,
            'level': com.current_level,
            'title': ""  # Populate from laws_dict if available
        })

        if is_last_layer:
            nodes.extend(get_graph_nodes_json(com.get_comm_node_positions(None),
                                              id, 
                                              com.current_level + 1,
                                              node_titles,
                                              color,
                                              extent))
    
    return communities_nodes, nodes


def get_community_edges_json(g_c, communities_dict):
    # Create community edges using community graph
    communities_edges = []
    for edge in g_c.edges():
        communities_edges.append({
            'source': communities_dict[edge[0]].uId,
            'target': communities_dict[edge[1]].uId,
            'weight': g_c.get_edge_data(*edge)['weight']
        })
    return communities_edges


def get_graph_nodes_json(node_positions, community_id, level, node_titles, color, extent):
    nodes = []
    for node, pos in node_positions.items():
        n_pos = get_normalize_pos(pos, extent)
        nodes.append({
            'id': node,
            'x': float(pos[0]),
            'y': float(pos[1]),
            'n_x': float(n_pos[0]),
            'n_y': float(n_pos[1]),
            'community': community_id,
            'radius': 2,
            'level': level,  
            'color': color,
            'title': node_titles.get(node, "")
        })
    return nodes


def get_graph_edges_json(unique_edges, citation_type):
    edges = []
    for source, target in unique_edges:
        edges.append({
            'source': source,
            'target': target,
            'type': citation_type.get(f"{source}_{target}", "citation")  # Default type as "citation"
        })
    return edges


def get_nodes_edges_json(graph, positions, community_dict, community_colors, node_titles):
    json_string = { 
        'nodes': [
            {
            'id': node,
            'x': float(positions[node][0]),
            'y': float(positions[node][1]),
            'community': community_dict[node],
            'radius': 2,  # Default radius if not specified
            'color': community_colors[community_dict[node]],
            'level': 3,
            'title': graph.nodes[node]['title']
            } for node in graph.nodes()
        ],
        'edges': [
            {'source': source, 'target': target} for source, target in graph.edges()
        ]

    }

    return json_string


def save_to_txt_file(fileName, nodes, isLayered = True):
    with open(fileName + ".txt", 'w') as f:
        for node in nodes:
            if isLayered:
                id = node['community'].split("_")[1]
                x = node['n_x']
                y = node['n_y']
            else:
                id = node['community']
                x = node['x']
                y = node['y']
            line = f"{node['id']} {x} {y} {id}"
            f.write(f"{line}\n")


def save_to_json_file(fileName, output_data):
    # Save output_data to a JSON file
    with open(fileName + ".json", "w") as json_file:
        json.dump(output_data, json_file, indent=4)


def generate_random_color(count):
    colors = []
    for i in range(count):
        colors.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
    return colors


def safe_str(value):
    return str(value) if value is not None else "0"


def output_json_generation(network, community_dict, node_titles, positions, fileName):

    community_colors = generate_random_color(len(network.communities))
    json_string = get_nodes_edges_json(network.graph, positions, community_dict, community_colors, node_titles)

    return get_output([], [], json_string['nodes'], json_string['edges'], fileName, isLayered= False)


def get_output(community_nodes, community_edges, nodes, edges, fileName, isLayered):
    # Final output structure
    output_data = {
        'communities': {
            'nodes': community_nodes,
            'edges': community_edges
        },
        'laws': {
            'nodes': nodes,
            'edges': edges
        }
    }

    save_to_txt_file(fileName, nodes, isLayered)
    save_to_json_file(fileName, output_data)

    return output_data


def get_normalize_pos(pos, old_extent, radius = None):
    # Get current extent
    min_x, min_y = old_extent[0]
    max_x, max_y = old_extent[1]

    # Define the target extent
    new_min, new_max = [-10, 10]

    scale_x = (new_max - new_min) / (max_x - min_x) if max_x != min_x else 1
    scale_y = (new_max - new_min) / (max_y - min_y) if max_y != min_y else 1
    normalized_x = new_min + (pos[0] - min_x) * scale_x
    normalized_y = new_min + (pos[1] - min_y) * scale_y

    if radius != None:
        return normalized_x, normalized_y, radius * min(scale_x, scale_y)
    
    return normalized_x, normalized_y


def convert_to_shapefile(json_communities_nodes, nodes, json_communities_edges, edges):
    # gdf_json_communities_nodes = nodes_to_gdf_with_buffer(json_communities_nodes, True)
    gdf_nodes = nodes_to_gdf_with_buffer(nodes)

    # gdf_json_communities_edges = edges_to_gdf(json_communities_edges, gdf_json_communities_nodes, True)
    gdf_edges = edges_to_gdf(edges, gdf_nodes)

    # Save to shapefiles
    # gdf_json_communities_nodes.to_file("shapefile/json_communities_nodes.shp")
    # gdf_json_communities_edges.to_file("shapefile/json_communities_edges.shp")
    gdf_nodes.to_file("shapefile/cosmo_nodes.shp")
    gdf_edges.to_file("shapefile/cosmo_edges.shp")


def nodes_to_gdf_with_buffer(nodes, radius = False):
    df = pd.DataFrame(nodes)

    if radius:
        gdf = gpd.GeoDataFrame(
            df, geometry=[Point(xy).buffer(radius) for xy, radius in zip(zip(df.x, df.y), df.radius)], crs="EPSG:4326"
        )
    else:
        gdf = gpd.GeoDataFrame(
            df, geometry=[Point(xy) for xy in zip(zip(df.x, df.y))], crs="EPSG:4326"
        )
    return gdf


# Convert edges to GeoDataFrame
def edges_to_gdf(edges, node_gdf, weight = False):
    edge_list = []
    for edge in edges:
        if not node_gdf[node_gdf['id'] == edge['source']].empty and not node_gdf[node_gdf['id'] == edge['target']].empty:
            source_node = node_gdf[node_gdf['id'] == edge['source']].iloc[0]
            target_node = node_gdf[node_gdf['id'] == edge['target']].iloc[0]
            line = LineString([(source_node.geometry.centroid.x, source_node.geometry.centroid.y),
                               (target_node.geometry.centroid.x, target_node.geometry.centroid.y)])
            
            if weight:
                edge_list.append({'source': edge['source'], 'target': edge['target'],  'weight': edge['weight'], 'geometry': line})
            else:
                edge_list.append({'source': edge['source'], 'target': edge['target'], 'geometry': line})
        else:
            print(f"Skipping edge with missing node: {edge}")
    edge_gdf = gpd.GeoDataFrame(edge_list, crs="EPSG:4326")
    return edge_gdf



# def get_radius(self):
#     radius = self.radius * min(self.scaling_factor_x, self.scaling_factor_y)
#     return radius


# def get_rescaled_pos(self, position, parent_pos, parent_radius):
#     parent_x_min = parent_pos[0] - parent_radius
#     parent_y_min = parent_pos[1] - parent_radius
#     return (parent_x_min + (position[0] - self.extent[0][0]) * self.scaling_factor_x, 
#                     parent_y_min + (position[1] - self.extent[0][1]) * self.scaling_factor_y) 