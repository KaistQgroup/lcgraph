import networkx as nx

import utils.utils as utils
from utils.lawdbcon import LawDB

from layered_community_aware_layout.layered_network import LayeredLayout
from spring_layout.spring_layout import SpringLayout



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








if __name__ == '__main__':
    grid_data = {}
    rescale_data = {}
    tsne_layout_data = {}
    spring_layout_data = {}
    layered_layout_data = {}

    article_spring_layout_data = {}

    law_db = LawDB(text_file = False, law_csv_file = False)
    law_graph = law_db.create_law_graph()
    

    get_layered_layout(law_graph)