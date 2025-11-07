from flask import Flask, jsonify, request
from flask_cors import CORS
import sys
import os
import multiprocessing
import networkx as nx
import json
import math

# This should be the absolute path to the directory
module_path = '/Users/marysm/Documents/programs/TSN/law_graph_layout_clean'
sys.path.append(module_path)

import utils.utils as utils
from utils.lawdbcon import LawDB

from layered_community_aware_layout.layered_network import LayeredLayout
from spring_layout.spring_layout import SpringLayout
from utils.evaluation import Metrics
# from bundling.force_directed_edge_bunding import apply_edge_bundling


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins


base_file_name = "legal_code_"

def get_layered_layout(law_graph):
    global layered_layout_data

    layout = LayeredLayout(law_graph)
    layout.layered_community_layout()

    layered_layout_data = utils.community_output_json_generation(layout.network, law_db, base_file_name + "lcgraph")


def get_tsne_layout():
    from tsne_layout.tsne_layout import TSNELayout

    global tsne_layout_data

    layout = TSNELayout(law_graph)
    community_dict, positions = layout.tsn_layout()

    print("t-SNE layout")
    cal_evalaution(layout.network, positions, community_dict)

    tsne_layout_data = utils.output_json_generation(layout.network, community_dict, law_db.node_titles, positions, base_file_name + "tsne")


def get_spring_layout():
    global spring_layout_data

    layout = SpringLayout(law_graph)
    community_dict, positions = layout.spring_law_layout()

    print("Spring Layout")
    cal_evalaution(layout.network, positions, community_dict)

    spring_layout_data = utils.output_json_generation(layout.network, community_dict, law_db.node_titles, positions, base_file_name + "spring")



def get_article_spring_layout():
    global article_spring_layout_data

    article_graph = law_db.create_article_graph()

    article_layout = SpringLayout(article_graph)
    community_dict, positions = article_layout.spring_law_layout()

    article_spring_layout_data = utils.output_json_generation(article_layout.network, community_dict, None, positions, "article_" + "spring")







def cal_evalaution(network, positions, community_dict, layered = False, original_positions = None):
    metrics = Metrics(network, positions, layered=layered, original_positions=original_positions)
    # print("Edge Crossings" + ": " + str(metrics.balaban_edge_crossings()))
    # print("Community Entropy" + ": " + str(metrics.community_entropy(community_dict)))
    # print("Group Overlap" + ": " + str(metrics.group_overlap(community_dict)))
    # print("Node Spread" + ": " + str(metrics.node_spread(community_dict)))
    # print("Node Occlusions" + ": " + str(metrics.node_occlusions()))
    # print("Spatial Autocorrelation" + ": " + str(metrics.spatial_autocorrelation(community_dict)))
    
    

@app.route('/layered_positions', methods=['GET'])
def get_layered_layout_json():
    return jsonify(layered_layout_data)



# @app.route('/layered_positions', methods=['GET'])
# def get_layered_layout_json():
#     offset = int(request.args.get('offset', 0))
#     limit = int(request.args.get('limit', 100))

#     # Paginate nodes and edges
#     paginated_data = {
#         "laws": {
#             "nodes": layered_layout_data["laws"]["nodes"][offset:offset+limit],
#             "edges": layered_layout_data["laws"]["edges"][offset:offset+limit]
#         },
#         "communities": {
#             "nodes": layered_layout_data["communities"]["nodes"][offset:offset+limit],
#             "edges": layered_layout_data["communities"]["edges"][offset:offset+limit]
#         }
#     }
    
    
#     return jsonify(paginated_data)



@app.route('/tsne_positions', methods=['GET'])
def get_tsne_layout_json():
    return jsonify(tsne_layout_data)

@app.route('/spring_positions', methods=['GET'])
def get_spring_layout_json():
    return jsonify(spring_layout_data)

@app.route('/articles_citations', methods=['GET'])
def get_articles_spring_layout_json():
    return jsonify(article_spring_layout_data)


@app.route('/contents_by_article/<int:article_id>', methods=['GET'])
def get_contents_by_article_id(article_id):
    contents = law_db.get_contents_by_article_id(article_id)
    return jsonify(contents)

@app.route('/articles_by_law/<int:law_id>', methods=['GET'])
def get_citations_by_law_id(law_id):
    print(law_id)

    graph = law_db.create_article_graph(law_id)
    with open("lcgraph_positions.json", "r") as file:
        laws_info = json.load(file)

    if graph:
        selected_laws = {
            node["id"]: node
            for node in laws_info["laws"]["nodes"] if node["id"] in graph.nodes
        }

        main_law_color = selected_laws.get(law_id)["color"]
        main_law_community = selected_laws.get(law_id)["community"]
    
        article_nodes = [node for node in graph.nodes if graph.nodes[node].get("type") == "article"]
        num_articles = len(article_nodes)
        sub_graph =  graph.subgraph(article_nodes)

        min_spacing = 50  
        initial_radius = (num_articles * min_spacing) / (2 * math.pi)
        circular_positions = nx.circular_layout(sub_graph, scale=initial_radius)

        max_distance = max(
            math.sqrt(pos[0] ** 2 + pos[1] ** 2)
            for pos in circular_positions.values()
        )

        radius = max_distance


        article_json_string = { 
            'nodes': [
                {
                'id': node,
                'x': float(circular_positions[node][0]),
                'y': float(circular_positions[node][1]),
                'community': main_law_community,
                'radius': 2,  # Default radius if not specified
                'color': main_law_color,
                'level': 3,
                'title': ''
                } for node in sub_graph
            ],
            'edges': [
                {'source': edge[0], 'target': edge[1], "weight": sub_graph.edges[edge].get("weight", 1)} for edge in sub_graph.edges()
            ]
        }

        
        central_position = (0, 0)
        positions = {law_id: central_position}

        for node in graph.nodes:
            if graph.nodes[node].get("type") == "law" and node != law_id:
                delta_x = selected_laws[node]["x"] - selected_laws[law_id]["x"]  
                delta_y = selected_laws[node]["y"] - selected_laws[law_id]["y"]
                distance = math.sqrt((delta_x ** 2 + delta_y ** 2))
                newDist = radius + distance + 600

                norm_x = delta_x / distance if distance != 0 else 0
                norm_y = delta_y / distance if distance != 0 else 0

                scaled_x = central_position[0] + newDist * norm_x
                scaled_y = central_position[1] + newDist * norm_y

                positions[node] = (scaled_x, scaled_y)
        
        laws = []
        for lnode in positions:
            info = selected_laws[lnode]
            info['x'] = positions[lnode][0]
            info['y'] = positions[lnode][1]
            laws.append(info)

        
        edges = [
                {'source': edge[0], 'target': edge[1], "weight": graph.edges[edge].get("weight", 1)} 
                for edge in graph.edges() if graph.edges[edge]['type']=="membership" or graph.edges[edge]['type']=="a-l-citation"
            ]

        response = {
            "articles": article_json_string,
            "laws": {'nodes': laws, 'edges': edges},
        }

        json_response = jsonify(response)
        json_response.headers.add("Access-Control-Allow-Origin", "*")
        return json_response
    
    default_response = jsonify({"articles": {'nodes': {}, 'edges': {}}, "laws": {'nodes': {}, 'edges': {}}})
    default_response.headers.add("Access-Control-Allow-Origin", "*")
    return default_response




if __name__ == '__main__':
    multiprocessing.set_start_method('fork')
    # Global data variable to store node positions and edges
    grid_data = {}
    rescale_data = {}
    tsne_layout_data = {}
    spring_layout_data = {}
    layered_layout_data = {}

    article_spring_layout_data = {}

    law_db = LawDB(text_file = False, law_csv_file = False)
    law_graph = law_db.create_law_graph()
    

    get_layered_layout(law_graph)
    # get_article_spring_layout()

    # get_spring_layout()
    # get_tsne_layout()
    app.run(debug=True, use_reloader=False)
