from __future__ import print_function

import numpy as np
import random
import json
import sys
import os

import networkx as nx
from networkx.readwrite import json_graph

version_info = list(map(int, nx.__version__.split('.')))
major = version_info[0]
minor = version_info[1]
assert (major <= 1) and (minor <= 11), "networkx major version > 1.11"

WALK_LEN = 5
N_WALKS = 50


def load_data(prefix, normalize=True, load_walks=False):
    graph_data = json.load(open(prefix + "-G.json"))
    graph = json_graph.node_link_graph(graph_data)
    if isinstance(graph.nodes()[0], int):
        def conversion(n):
            return int(n)
    else:
        def conversion(n):
            return n

    if os.path.exists(prefix + "-feats.npy"):
        feats = np.load(prefix + "-feats.npy")
    else:
        print("No features present.. Only identity features will be used.")
        feats = None
    id_map = json.load(open(prefix + "-id_map.json"))
    id_map = {conversion(k): int(v) for k, v in id_map.items()}
    walks = []
    class_map = json.load(open(prefix + "-class_map.json"))
    if isinstance(list(class_map.values())[0], list):
        def lab_conversion(n):
            return n
    else:
        def lab_conversion(n):
            return int(n)

    class_map = {conversion(k): lab_conversion(v) for k, v in class_map.items()}

    # Remove all nodes that do not have val/test annotations
    # (necessary because of networkx weirdness with the Reddit data)
    broken_count = 0
    for node in graph.nodes():
        if 'val' not in graph.node[node] or 'test' not in graph.node[node]:
            graph.remove_node(node)
            broken_count += 1
    print("Removed {:d} nodes that lacked proper annotations due to networkx versioning issues".format(broken_count))

    # Make sure the graph has edge train_removed annotations
    # (some datasets might already have this..)
    print("Loaded data.. now preprocessing..")
    for edge in graph.edges():
        if (graph.node[edge[0]]['val'] or graph.node[edge[1]]['val'] or
                graph.node[edge[0]]['test'] or graph.node[edge[1]]['test']):
            graph[edge[0]][edge[1]]['train_removed'] = True
        else:
            graph[edge[0]][edge[1]]['train_removed'] = False

    if normalize and feats is not None:
        from sklearn.preprocessing import StandardScaler
        train_ids = np.array([id_map[n] for n in graph.nodes()
                              if not graph.node[n]['val'] and not graph.node[n]['test']])
        train_feats = feats[train_ids]
        scaler = StandardScaler()
        scaler.fit(train_feats)
        feats = scaler.transform(feats)

    if load_walks:
        with open(prefix + "-walks.txt") as f:
            for line in f:
                walks.append(map(conversion, line.split()))

    return graph, feats, id_map, walks, class_map


def run_random_walks(graph, nodes, num_walks=N_WALKS):
    pairs = []
    for count, node in enumerate(nodes):
        if graph.degree(node) == 0:
            continue
        for i in range(num_walks):
            curr_node = node
            for j in range(WALK_LEN):
                next_node = random.choice(graph.neighbors(curr_node))
                # self co-occurrences are useless
                if curr_node != node:
                    pairs.append((node, curr_node))
                curr_node = next_node
        if count % 1000 == 0:
            print("Done walks for", count, "nodes")
    return pairs


if __name__ == "__main__":
    """ Run random walks """
    graph_file = sys.argv[1]
    out_file = sys.argv[2]
    G_data = json.load(open(graph_file))
    G = json_graph.node_link_graph(G_data)
    nodes = [n for n in G.nodes() if not G.node[n]["val"] and not G.node[n]["test"]]
    G = G.subgraph(nodes)
    pairs = run_random_walks(G, nodes)
    with open(out_file, "w") as fp:
        fp.write("\n".join([str(p[0]) + "\t" + str(p[1]) for p in pairs]))
