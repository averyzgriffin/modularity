import networkx as nx
from networkx.algorithms.community.centrality import girvan_newman
import matplotlib.pyplot as plt
import numpy as np
import json
from data_viz import visualize_solo_network
from network_graphs import NetworkGraph, visualize_graph_data
from modularity import compute_modularity


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
    for i in range(1):
        w_file = open(f"saved_weights/network_{i}.json", "r")
        network = json.load(w_file)
        the_hard_way(network)
        population.append(network)
        w_file.close()
    return population


population = load_networks()

# convert_networks(population[:10], runname="test", gen=42)

ng = NetworkGraph(population[0])
ng.convert2graph()
ng.get_data()
# ng.draw_graph("", show=True)

Q = compute_modularity(ng)



x = 4


# # Convert network to graph
# G = nx.Graph()
#
# input = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8"]
# l1 = ["L1.1", "L1.2", "L1.3", "L1.4", "L1.5", "L1.6", "L1.7", "L1.8"]
# l2 = ["L2.1", "L2.2", "L2.3", "L2.4"]
# l3 = ["L3.1", "L3.2"]
# output = ["output"]
#
# # Adding input layer nodes and connections to 1st layer
# for i in range(len(input)):
#     G.add_nodes_from([(input[i], {'activation': 1, 'pos': (0, 35*(i+1))})])
#     for j in range(len(l1)):
#         if network['thetas'][0][i][j] != 0:
#             G.add_edges_from([(input[i], l1[j], {'weight': network['thetas'][0][i][j] })])
# # Adding 1st layer nodes and connections to 2nd layer
# for k in range(len(l1)):
#     G.add_nodes_from([(l1[k], {'activation': network['thresholds'][0][k], 'pos': (150, 35*(k+1))})])
#     for l in range(len(l2)):
#         if network['thetas'][1][k][l] != 0:
#             G.add_edges_from([(l1[k], l2[l], {'weight': network['thetas'][1][k][l] })])
# # Adding 2nd layer nodes and connections to 3rd layer
# for m in range(len(l2)):
#     G.add_nodes_from([(l2[m], {'activation': network['thresholds'][1][m], 'pos': (300, 35*(m+3))})])
#     for n in range(len(l3)):
#         if network['thetas'][2][m][n] != 0:
#             G.add_edges_from([(l2[m], l3[n], {'weight': network['thetas'][2][m][n] })])
# # Adding 3rd layer nodes and connections to output layer
# for o in range(len(l3)):
#     G.add_nodes_from([(l3[o], {'activation': network['thresholds'][2][o], 'pos': (450, 35*(o+4))})])
#     for p in range(len(output)):
#         if network['thetas'][3][o][p] != 0:
#             G.add_edges_from([(l3[o], output[p], {'weight': network['thetas'][3][o][p] })])
# # Adding output layer nodes
# G.add_nodes_from([(output[0], {'activation': network['thresholds'][3][0], 'pos': (550, 160)})])
#
# # Mapping edge weights to colors
# edges = G.edges()
# edge_color_map = {-2: 'darkred', -1: 'r', 0: 'pink', 1: 'c', 2: 'b'}
# edge_colors = [edge_color_map[G[u][v]['weight']] for u,v in edges]
#
# # Defining the positions of each node
# node_pos = {}
# for n in G.nodes:
#     node_pos[n] =  G.nodes[n]['pos']
#
# stop = 1
#
# # nx.draw(G, pos=node_pos, edge_color=edge_colors, node_size=1200, with_labels=True)
# # plt.show()
#
# communities = girvan_newman(G)
# node_groups = []
# for com in next(communities):
#     node_groups.append(list(com))
#
# print(node_groups)
# print(len(node_groups))
#
# module_color_map = ['orange', 'lime', 'k', 'r', 'yellow']
# module_colors = []
# for node in G.nodes:
#     for c in range(len(node_groups)):
#         if node in node_groups[c]:
#             module_colors.append(module_color_map[c])
#             break
#
#
# # nx.draw(G, pos=node_pos, edge_color=module_colors, node_size=1200, with_labels=True)
# # plt.show()
#
# fig = plt.figure(figsize=(16,8))
# ax1 = fig.add_subplot(1, 2, 1)
# ax2 = fig.add_subplot(1, 2, 2)
# nx.draw(G, pos=node_pos, edge_color=edge_colors, node_size=1200, with_labels=True, ax=ax1)
# nx.draw(G, pos=node_pos, edge_color=edge_colors, node_color=module_colors, node_size=1200, with_labels=True, ax=ax2)
# # ax1.set_title('Best Loss Each Generation')
# # ax1.legend()
# # ax2.set_xlabel('Generation (n)')
# # ax2.set_ylabel('Loss')
# # ax2.set_title('Average Loss Each Generation')
# # ax2.legend()
# plt.show()


