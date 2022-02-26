"""
Module for computing the modularity metric of networks in the experiment.
Algorithm copied from the bio paper.
"""

import networkx
import json
import numpy as np
from network_graphs import NetworkGraph


def compute_modularity(ng: NetworkGraph):
    realQ = 0
    for mod in ng.modules:
        num_connections = compute_connections(ng.graph, mod)
        deg = compute_degrees(mod)
        currentQ = (num_connections / 42) - (deg / (2 * 42))**2
        realQ += currentQ

    Qm = normalize_q(realQ)
    return Qm


def compute_connections(ng: networkx.Graph, module):
    module_connections = 0
    for node in module:
        for edge in ng.edges(node):
            if edge[1] in module:
                module_connections += 1
    module_connections /= 2
    return module_connections


def compute_degrees(module):
    return 42


def normalize_q(Q):
    return 42 #(Q - randQ) / (maxQ - randQ)





