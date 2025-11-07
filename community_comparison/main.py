import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import time
from profiling import time_mem_section, write_csv_row, _rss_mb
from utils.lawdbcon import LawDB
import hsbm_partition as HSBM
import louvain_hierarchy as LH
from modularity_utils import write_modularity_long_csv
import os, csv, time
from typing import Dict, Any, List

from performance_comparison import run_all_benchmarks, write_results_csv
from modularity_eval import run_modularity_methods_and_collect, write_modularity_csv

from network_module import LayeredLawNetwork
from hsbm_network_module import HSBMLayeredLawNetwork
from ls_network_module import LSLayeredLawNetwork
from h_louvain_network_module import HLouvainLayeredLawNetwork
from leiden_network_module import LeidenLayeredLawNetwork

import tracemalloc
import time


law_db = LawDB(text_file = True, law_csv_file = False)
law_graph = law_db.create_law_graph()

# lcgraph = LayeredLawNetwork(law_graph)
# leiden_graph = LeidenLayeredLawNetwork(law_graph)
# hsbm = HSBMLayeredLawNetwork(law_graph)
# ls_graph = LSLayeredLawNetwork(law_graph)
# hlouvain_graph = HLouvainLayeredLawNetwork(law_graph)


# benchmarks = [
#     ("LCGraph(init)",        lambda G: LayeredLawNetwork(G)),
#     ("Leiden(init)",         lambda G: LeidenLayeredLawNetwork(G)),
#     ("HSBM(init)",           lambda G: HSBMLayeredLawNetwork(G)),
#     ("LS(init)",             lambda G: LSLayeredLawNetwork(G)),
#     ("HLouvain(init)",       lambda G: HLouvainLayeredLawNetwork(G)),
# ]

# results = run_all_benchmarks(law_graph, benchmarks)
# write_results_csv(results, filepath="wiki_time_memory_results.csv", overwrite=True)
# print("Wrote", len(results), "rows to bench_results.csv")



benchmarks = [
    ("LCGraph(init)",        lambda G: LayeredLawNetwork(G)),
    ("Leiden(init)",         lambda G: LeidenLayeredLawNetwork(G)),
    ("HSBM(init)",           lambda G: HSBMLayeredLawNetwork(G)),
    ("LS(init)",             lambda G: LSLayeredLawNetwork(G)),
    ("HLouvain(init)",       lambda G: HLouvainLayeredLawNetwork(G)),
]

rows = run_modularity_methods_and_collect(
    law_graph,
    benchmarks,
    weight="auto",           # or None / "weight"
    resolution=1.0,          # modularity γ (not CPM γ)
    max_level=None,          # or an int to cap levels
    fill_missing_mode="remainder",  # preferred for stable Q
)
write_modularity_csv(rows, "wiki_modularity_by_level.csv", overwrite=True)
print("Wrote", len(rows), "rows → modularity_by_level.csv")
