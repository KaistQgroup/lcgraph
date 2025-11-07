import mysql.connector
import networkx as nx
import pandas as pd
from networkx.algorithms.community import louvain_communities
import json
import random


class LawDB:
    def __init__(self, text_file = False, law_csv_file = False):
        self.data = []

        if text_file:
            self.load_txt_file()
        elif law_csv_file:
            self.load_csv()
        else:
            self.fetch_law_level_citation_data()            

    def get_db_connection(self):
        # Only initialize the connection pool if using the database
        db1_config = {
            'host': '',
            'user': '',
            'password': '',
            'database': ''
        }
        # Create a connection pool for better performance
        self.pool = mysql.connector.pooling.MySQLConnectionPool(
            pool_name="mypool",
            pool_size=5,
            **db1_config
        )

        return self.pool.get_connection()

    
    def fetch_data(self, query, values=None):
        conn = self.get_db_connection()
        with conn.cursor() as cursor:
            if values:
                cursor.execute(query, values)  # Parameterized query
            else:
                cursor.execute(query)  # Simple query
            rows = cursor.fetchall()
        conn.close()
        return rows
    

    def fetch_law_level_citation_data(self):

        sql = """
            """

        rows = self.fetch_data(sql)
        column_names = ['citor_law_id', 'citor_law_title', 'citee_law_id', 'citee_law_title']
        self.data = [dict(zip(column_names, row)) for row in rows] 
        # df.to_csv('law_data.csv', index=False)  


    def load_csv(self):
        """Load large CSV data using Dask for better performance."""
        csv_path = "law_data.csv"
        self.data = pd.read_csv(csv_path, encoding='latin1').to_dict(orient='records')


    def load_txt_file(self):
        """Load citation data from a .txt file."""
        txt_file = "wiki_edges.txt"
        with open(txt_file, 'r') as file:
            next(file) 
            for line in file:
                parts = line.strip().split()
                if len(parts) >= 2:
                    source = int(parts[0])
                    target = int(parts[1])
                    self.data.append({
                        'citor_law_id': source,
                        'citee_law_id': target,
                        'citor_law_title': "",
                        'citee_law_title': "",
                        "citation_type": 1
                    })

        
    def create_law_graph(self):
        """Create a NetworkX graph from citation data."""
        G = nx.Graph()
        added_nodes = set()
        self.unique_edges = set()
        self.node_titles = {}
        self.citation_type = {}

        for row in self.data:
            source = row['citor_law_id']
            target = row['citee_law_id']
            source_title = row['citor_law_title']
            target_title = row['citee_law_title']

            self.citation_type[f"{source}_{target}"] = 1 #row['citation_type']

            if source != target:
                if source not in added_nodes:
                    G.add_node(source, title=self.node_titles.setdefault(source, source_title))
                    added_nodes.add(source)

                if target not in added_nodes:
                    G.add_node(target, title=self.node_titles.setdefault(target, target_title))
                    added_nodes.add(target)

                if (source, target) not in self.unique_edges and (target, source) not in self.unique_edges:
                    G.add_edge(source, target)
                    self.unique_edges.add((source, target))

        print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
        return G
    