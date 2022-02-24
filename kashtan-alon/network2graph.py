import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import json


def the_hard_way(network):
    keys = ["thetas", "thresholds"]
    for key in keys:
        for i in range(len(network[key])):
            for j in range(len(network[key][i])):
                if isinstance(network[key][i][j], list):
                    network[key][i][j] = np.array(network[key][i][j])
            if isinstance(network[key][i], list):
                network[key][i] = np.array(network[key][i])


def load_networks():
    population = []
    for i in range(100):
        w_file = open(f"saved_weights/network_{i}.json", "r")
        network = json.load(w_file)
        the_hard_way(network)
        population.append(network)
        w_file.close()
    return population


# Convert network to graph
G = nx.DiGraph()

input = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8"]
l1 = ["n1l1", "n2l1", "n3l1", "n4l1", "n5l1", "n6l1", "n7l1", "n8l1"]
l2 = ["n1l2", "n2l2", "n3l2", "n4l2"]
l3 = ["n1l3", "n2l3"]
output = ["output"]

# Adding input layer nodes and connections to 1st layer
for i in range(len(input)):
    G.add_nodes_from([(input[i], {'activation': 1, 'pos': (0, 50*(i+1))})])
    for j in l1:
        G.add_edges_from([(input[i], j, {'weight': np.random.randint(-2,3)})])
# Adding 1st layer nodes and connections to 2nd layer
for k in range(len(l1)):
    G.add_nodes_from([(l1[k], {'activation': 1, 'pos': (150, 50*(k+1))})])
    for l in l2:
        G.add_edges_from([(l1[k], l, {'weight': np.random.randint(-2,3)})])
# Adding 2nd layer nodes and connections to 3rd layer
for m in range(len(l2)):
    G.add_nodes_from([(l2[m], {'activation': 1, 'pos': (300, 50*(m+3))})])
    for n in l3:
        G.add_edges_from([(l2[m], n, {'weight': np.random.randint(-2,3)})])
# Adding 3rd layer nodes and connections to output layer
for o in range(len(l3)):
    G.add_nodes_from([(l3[o], {'activation': 1, 'pos': (450, 50*(o+4))})])
    for p in output:
        G.add_edges_from([(l3[o], p, {'weight': np.random.randint(-2,3)})])
# Adding output layer nodes
G.add_nodes_from([(output[0], {'activation': 1, 'pos': (550, 250)})])

# Mapping edge weights to colors
edges = G.edges()
edge_color_map = {-2: 'darkred', -1: 'r', 0: 'w', 1: 'c', 2: 'b'}
edge_colors = [edge_color_map[G[u][v]['weight']] for u,v in edges]

# Defining the positions of each node
node_pos = {}
for n in G.nodes:
    node_pos[n] =  G.nodes[n]['pos']

stop = 1

nx.draw(G, pos=node_pos, edge_color=edge_colors, with_labels=True)
plt.show()

