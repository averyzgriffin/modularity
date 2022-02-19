"""
Functions for writing and plotting Graphviz files. Used to visualize the neural networks
"""

# Original author
# Madhavun Candadai

# Inspired by
# https://tgmstat.wordpress.com/2013/06/12/draw-neural-network-diagrams-graphviz/


import graphviz
from graphviz import Source
import sys


def write_graphviz(network, file_path):
    layers = [8, 8, 4, 2, 1]

    layers_str = ["Input"] + ["L1"] + ["L2"] + ["L3"] + ["Output"]
    layers_col = ["none"] + ["none"] * (len(layers) - 2) + ["none"]
    layers_fill = ["black"] + ["blue"] + [ "red"] + ["green"] + ["gray"]

    penwidth = 15
    font = "Hilda 10"

    orig_stdout = sys.stdout
    f = open(file_path, "w")
    sys.stdout = f

    print("digraph G {")
    print("\tordering=\"in\";")
    print("\tfontname = \"{}\"".format(font))
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
                print("l{}{} [label={}]".format(i + 1, a, network["thresholds"][i-1][a][0]), end=' ')

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
    dot.render(view=False)












