"""
Class for automatically converting neural networks into graphs.
Also detects communities (modules) and plots color-coded graphs based on modularity.
"""

import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.community.centrality import girvan_newman
from os import makedirs
from os.path import join
from tqdm import tqdm


class NetworkGraph:

    def __init__(self, network):

        self.network = network

        self.graph = self.initiate_graph()

        self.input = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8"]
        self.l1 = ["L1.1", "L1.2", "L1.3", "L1.4", "L1.5", "L1.6", "L1.7", "L1.8"]
        self.l2 = ["L2.1", "L2.2", "L2.3", "L2.4"]
        self.l3 = ["L3.1", "L3.2"]
        self.output = ["output"]
        self.layers = [self.l1, self.l2, self.l3, self.output]

        self.edge_color_map = {-2: 'darkred', -1: 'r', 0: 'pink', 1: 'c', 2: 'b'}
        self.edge_colors = []
        self.modules = []
        self.module_color_map = ['orange', 'lime', 'k', 'r', 'y', 'b', 'm', 'gray',
                                 'c', 'g', 'olive', 'pink', 'brown', 'indigo', 'lightgray'
                                 'teal', 'palegreen', 'skyblue', 'palegoldenrod']
        self.module_colors = []
        self.node_pos = {}

    @staticmethod
    def initiate_graph():
        return nx.Graph()

    def convert2graph(self):
        # Convert input layer to graph
        for i in range(len(self.input)):
            self.graph.add_nodes_from([(self.input[i], {'activation': 1, 'pos': (0, 35 * (i + 1))})])
            for j in range(len(self.l1)):
                if self.network['thetas'][0][i][j] != 0:
                    self.graph.add_edges_from([(self.input[i], self.l1[j], {'weight': self.network['thetas'][0][i][j]})])

        # Convert hidden layers to graph
        for l in range(len(self.layers)-1):
            for i in range(len(self.layers[l])):
                self.graph.add_nodes_from([(self.layers[l][i], {'activation': self.network['thresholds'][0][i], 'pos': (160*(l+1), 35 * (i + 1 + (1.3*l)))})])
                for j in range(len(self.layers[l+1])):
                    if self.network['thetas'][l+1][i][j] != 0:
                        self.graph.add_edges_from([(self.layers[l][i], self.layers[l+1][j], {'weight': self.network['thetas'][l+1][i][j]})])

        # Convert output layer to graph
        self.graph.add_nodes_from([(self.output[0], {'activation': self.network['thresholds'][len(self.layers)-1][0], 'pos': (650, 160)})])

    def get_data(self):
        self.compute_edge_colors()
        self.compute_node_pos()
        self.compute_communities()
        self.compute_module_colors()

    def compute_edge_colors(self):
        edges = self.graph.edges()
        self.edge_colors = [self.edge_color_map[self.graph[u][v]['weight']] for u, v in edges]

    def compute_node_pos(self):
        for n in self.graph.nodes:
            self.node_pos[n] = self.graph.nodes[n]['pos']

    def compute_communities(self):
        communities = girvan_newman(self.graph)
        for com in next(communities):
            self.modules.append(list(com))

    def compute_module_colors(self):
        for node in self.graph.nodes:
            for m in range(len(self.modules)):
                if node in self.modules[m]:
                    self.module_colors.append(self.module_color_map[m])
                    break

    def draw_graph(self, file_path, show=False):
        fig = plt.figure(figsize=(14, 7))
        ax1 = fig.add_subplot(1, 1, 1)
        nx.draw(self.graph, pos=self.node_pos, edge_color=self.edge_colors, node_color=self.module_colors, node_size=800, with_labels=True,
                ax=ax1)
        ax1.set_title('Graph color coded by detected modules')

        plt.text(.90, .90, f"Loss: {self.network['''loss''']}", transform=ax1.transAxes)

        if self.network["best"] == "True":
            plt.text(.90, .80, "*Best Network in Generation", transform=ax1.transAxes)

        if show: plt.show()

        # def save_graph(self):
        plt.savefig(file_path + ".png")

        fig.clear()
        plt.close(fig)


# Wrapper function to send networks in a population through networkx graph converter
def visualize_graph_data(population, runname, gen):
    for i in tqdm(range(len(population)), desc="Converting networks to graphs"):
        makedirs(join('networkx_graphs', runname, f"gen_{gen}").replace("\\", "/"), exist_ok=True)
        file_path = join('networkx_graphs', runname, f"gen_{gen}", f'network_{i}').replace("\\", "/")
        ng = NetworkGraph(population[i])
        ng.convert2graph()
        ng.get_data()
        ng.draw_graph(file_path)

