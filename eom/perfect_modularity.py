import os
os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'
from os.path import join
from os import makedirs
import time

import numpy as np
from tqdm import tqdm

from graphviz import Source
import sys

from graph import NetworkGraph, visualize_graph_data
from modularity import compute_modularity, normalize_q

from main import evaluate_population, generate_samples


def build_perfect_network():
    theta0 = np.array([
                        [1,1,0,0,0,0,0,0],
                        [1,1,0,0,0,0,0,0],
                        [-1,1,0,0,0,0,0,0],
                        [-1,1,0,0,0,0,0,0],
                        [0,0,0,0,0,0,1,-1],
                        [0,0,0,0,0,0,1,-1],
                        [0,0,0,0,0,0,1,1],
                        [0,0,0,0,0,0,1,1]])
    theta1 = np.array([[1,0,0,0],
                       [1,0,0,0],
                       [0,0,0,0],
                       [0,0,0,0],
                       [0,0,0,0],
                       [0,0,0,0],
                       [0,0,0,1],
                       [0,0,0,1]])
    theta2 = np.array([[0,1],
                       [0,0],
                       [0,0],
                       [0,1]])
    theta3 = np.array([[0],
                       [1]])

    bias0 = np.array([[0],[2],[10],[10],[10],[10],[2],[0]])
    bias1 = np.array([[0],[10],[10],[0]])
    bias2 = np.array([[10],[1]])
    bias3 = np.array([[0]])

    thetas = [theta0, theta1, theta2, theta3]
    biases = [bias0, bias1, bias2, bias3]

    network = {"thetas": thetas, "biases": biases, "loss": 0, "q": "n/a"}
    return network


def write_graphviz(network, file_path):
    # layers = [8, 8, 4, 2, 1]
    layers = []
    for theta in network["thetas"]:
        layers.append(len(theta))
    layers.append(1)

    layers_str = ["Input"] + ["L1"] + ["L2"] + ["L3"] + ["Output"]
    layers_col = ["none"] + ["none"] * (len(layers) - 2) + ["none"]
    layers_fill = ["black"] + ["blue"] + [ "red"] + ["green"] + ["gray"]

    penwidth = 15

    orig_stdout = sys.stdout
    f = open(file_path, "w")
    sys.stdout = f

    print("digraph G {")
    print("\tordering=\"in\";")
    print("\trankdir=LR")
    print("\tsplines=line")
    print("\tnodesep=.08;")
    print("\tranksep=1;")
    print("\tedge [color=black, arrowsize=.5];")
    print("\tnode [fixedsize=true,label=\"\",style=filled," + \
        "color=none,fillcolor=gray,shape=circle,ordering=\"in\"]\n")

    # Clusters
    for i in range(0, len(layers)):
        print(("\tsubgraph cluster_{} {{".format(i)))
        print("\t\tordering=\"in\";")
        print(("\t\tcolor={};".format(layers_col[i])))
        print(("\t\tnode [style=filled, color=white, penwidth={},"
              "fillcolor={} shape=circle,ordering=\"in\"];".format(
                  penwidth,
                  layers_fill[i])))

        print(("\t\t"), end=' ')

        if i == 0:
            for a in range(layers[i]):
                print("l{}{} [label=x_{}]".format(i + 1, a, a), end=' ')
        else:
            for a in range(layers[i]):
                print("l{}{} [label={}]".format(i + 1, a, network["biases"][i-1][a][0]), end=' ')

        print(";")
        print(("\t\tlabel = {};".format(layers_str[i])))

        print("\t}\n")

    # Nodes
    for i in range(1, len(layers)):
        for a in range(layers[i - 1]):
            for b in range(layers[i]):
                temp = network["thetas"][i-1][a][b]
                if network["thetas"][i-1][a][b] == 1:
                    print("\tl{}{} -> l{}{} [color=\"cyan\"]".format(i, a, i + 1, b))
                elif network["thetas"][i-1][a][b] == 2:
                    print("\tl{}{} -> l{}{} [color=\"cyan4\"]".format(i, a, i + 1, b))
                elif network["thetas"][i-1][a][b] == -1:
                    print("\tl{}{} -> l{}{} [color=\"red\"]".format(i, a, i + 1, b))
                elif network["thetas"][i-1][a][b] == -2:
                    print("\tl{}{} -> l{}{} [color=\"red4\"]".format(i, a, i + 1, b))

    print("}")

    sys.stdout = orig_stdout
    f.close()


def plot_graphviz(file_path):
    dot = Source.from_file(file_path)
    dot.render(view=True)


def visualize_solo_network(network, name):
    makedirs('graphviz', exist_ok=True)
    file_path = join('graphviz', f"{name}.txt").replace("\\", "/")
    write_graphviz(network, file_path)
    plot_graphviz(file_path)


def evaluate_q(population, normalize, graph=NetworkGraph):
    population_q = []
    for network in tqdm(population, desc="Computing modularity for networks in population"):
        network["q"] = 0
        ng = graph(network)
        ng.convert2graph()
        ng.get_data()
        qvalue = compute_modularity(ng)
        if normalize:
            qvalue = normalize_q(qvalue)
        network["q"] = round(qvalue, 3)
        population_q.append(network["q"])
    return population_q


perfect_network = build_perfect_network()
# visualize_solo_network(perfect_network, f"perfect_network_{time.time()}")
# visualize_graph_data([perfect_network], f"perfect_network_{time.time()}", 0)
# q = evaluate_q([perfect_network], normalize=True)
# print(q)

samples = generate_samples()
and_loss = evaluate_population([perfect_network], samples, "loss", True, "tanh")
or_loss = evaluate_population([perfect_network], samples, "loss", False, "tanh")

print("and ", and_loss)
print("or ", or_loss)
















