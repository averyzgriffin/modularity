"""
Module for computing the modularity metric of networks in the experiment.
Algorithm copied from the bio paper.
"""

import networkx
import json
import numpy as np
from network_graphs import NetworkGraph
from tqdm import tqdm
# from neural_network import build_network, apply_neuron_constraints


def compute_modularity(ng: NetworkGraph):
    realQ = 0
    total_edges = len(ng.graph.edges)
    for mod in ng.modules:
        num_connections = compute_connections(ng.graph, mod)
        deg = compute_degrees(ng.graph, mod)
        current_q = compute_q(num_connections, total_edges, deg)
        realQ += current_q

    return realQ


def compute_q(ls, L, ds):
         # (num_connections / total_edges) - (deg / (2 * total_edges))**2
    return (ls / L) - (ds / (2*L))**2


def compute_connections(G: networkx.Graph, module):
    module_connections = 0
    for node in module:
        for edge in G.edges(node):
            if edge[1] in module:
                module_connections += 1
    module_connections /= 2
    return module_connections


def compute_degrees(G: networkx.Graph, module):
    deg = 0
    for node in module:
        deg += G.degree()[node]
    return deg


def normalize_q(Q):
    randQ = .2011
    maxQ = .82
    return (Q - randQ) / (maxQ - randQ)


def compute_randQ(random_population):
    totalQ = 0
    for network in tqdm(random_population, desc="Computing Modularity"):
        ng = NetworkGraph(network)
        ng.convert2graph()
        ng.get_data()
        totalQ += compute_modularity(ng)

    averageQ = totalQ / len(random_population)
    return averageQ


