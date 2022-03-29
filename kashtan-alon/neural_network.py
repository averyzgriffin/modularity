"""
Contains functions related to the neural networks themselves
E.g. building the neural networks, computing feed-forward, thresholding, loss-functions.
This is called by the main.py module
"""

import random

import numpy as np
from tqdm import tqdm
import yaml

from graphs import NetworkGraph
from modularity import compute_modularity, normalize_q


class NeuralNetwork:

    def __init__(self):
        
        self.thetas = []
        self.thresholds = []
        self.loss = 0
        self.performance_loss = 0
        self.connection_loss = 0
        self.performance_weight = 2
        self.best = "False"
        self.q = 0

        self.both_count = 0
        self.left_count = 0
        self.right_count = 0
        self.none_count = 0

        self.loss_func = self.original_loss
        # self.threshold_output = True
        # self.fixate_thresholds = False
        # self.fixate_final_weights = False
        # self.weight_domain = [-1,1]
        # self.load_config()

        self.build_network()

    @staticmethod
    def load_config():
        with open("experiment.yaml", 'r') as file:
            config = yaml.safe_load(file)

    def build_network(self):
        theta1 = np.random.randint(-2,2, (8,8))
        theta2 = np.random.randint(-2,2, (8,4))
        theta3 = np.random.randint(-2,2, (4,2))
        theta4 = np.random.randint(-2,2, (2,1))
        # theta4 = np.ones((2,1))
        thrsh1 = np.random.randint(-4,3, (8,1))
        thrsh2 = np.random.randint(-4,3, (4,1))
        thrsh3 = np.random.randint(-4,3, (2,1))
        thrsh4 = np.random.randint(-2,1, (1,1))

        self.thetas = [theta1, theta2, theta3, theta4]
        self.thresholds = [thrsh1, thrsh2, thrsh3, thrsh4]

        self.apply_neuron_constraints()
        # self.network = {"thetas": self.thetas, "thresholds": self.thresholds, "loss": 0}

    def evaluate_network(self, sample, goal_is_and):
        x = sample["pixels"]

        # prediction = self.feed_forward_tanh(x)
        prediction = self.feed_forward(x)

        loss = self.calculate_loss(prediction, sample, goal_is_and)
        if loss != 0:
            if sample["int_label"] == 3:
                self.both_count += 1
            if sample["int_label"] == 1:
                self.left_count += 1
            if sample["int_label"] == 2:
                self.right_count += 1
            if sample["int_label"] == 0:
                self.none_count += 1

        # self.loss += loss
        self.performance_loss += loss

    def feed_forward(self, x):
        thetas = self.thetas
        ts = self.thresholds
        for i in range(len(thetas)):
            if i == 0:
                z = np.dot(x.transpose(), thetas[i])
            else:
                z = np.dot(z, thetas[i])
    
            if i != len(thetas)-1: # TODO this is a luc thing
                self.apply_threshold(z, ts[i])
        return z

    def feed_forward_tanh(self, x):
        thetas = self.thetas
        ts = self.thresholds
        for i in range(len(thetas)):
            if i == 0:
                z = np.dot(x.transpose(), thetas[i])
            else:
                z = np.dot(z, thetas[i])

            a = np.sum([z.reshape(len(z),1), ts[i]], axis=0)
            z = self.apply_tanh(a).reshape(len(a))

        if z >= 0: z = 1
        else: z = 0
        return z

    def apply_tanh(self, z):
        a = np.tanh(20*z)
        return a

    @staticmethod
    def apply_threshold(z, t):
        for i in range(len(z)):
            if z[i] > t[i]:
                z[i] = 1
            else:
                z[i] = 0

    def calculate_loss(self, prediction, sample, goal_is_and):
        loss = self.original_loss(prediction, sample, goal_is_and)
        # loss = self.LandR_vs_LorR_loss(prediction, sample, goal_is_and)
        return loss

    @staticmethod
    def original_loss(prediction, sample, goal_is_and):
        if goal_is_and:
            if sample["int_label"] == 3: label = 1
            else: label = 0
        elif not goal_is_and:
            if sample["int_label"] == 1 or sample["int_label"] == 2 or sample["int_label"] == 3:
                label = 1
            else:
                label = 0

        if prediction - label != 0:
            stop = 3

        return int((prediction - label) ** 2)

    @staticmethod
    def LandR_vs_LorR_loss(prediction, sample, goal_is_and=False):
        if sample["int_label"] == 3: label = 2
        elif sample["int_label"] == 1 or sample["int_label"] == 2: label = 1
        else: label = 0

        return int((prediction - label)**2)
    
    def apply_neuron_constraints(self):
        thetas = self.thetas
        for theta in thetas:
            theta = theta.transpose()
            for node_num in range(len(theta)):
                total = sum(abs(theta[node_num]))
                while total > 3:
                    choice = random.randint(0, len(theta.transpose()) - 1)
                    if theta[node_num][choice] > 0:
                        theta[node_num][choice] -= 1
                    elif theta[node_num][choice] < 0:
                        theta[node_num][choice] += 1
                    total = sum(abs(theta[node_num]))

    def calculate_connection_loss(self):
        self.connection_loss = sum([sum(1 for w in theta.flatten() if w != 0) for theta in self.thetas])

    def normalize_losses(self, num_samples, max_connections):
        self.performance_loss = round(self.performance_loss / num_samples, 3)
        self.connection_loss = round(self.connection_loss / max_connections, 3)

    def combine_losses(self):
        # self.loss = round((self.performance_weight * self.performance_loss + self.connection_loss) \
        #             / (self.performance_weight + 1), 3)
        self.loss = self.performance_loss

class LucNeuralNetwork(NeuralNetwork):
    
    def __init__(self):
        super().__init__()

    def build_network(self):
        theta1 = np.random.choice([0, 1], (8, 2))
        theta2 = np.random.choice([0, 1], (2, 1))
        thrsh1 = np.ones((2, 1))
        thrsh2 = np.ones((1, 1))
    
        self.thetas = [theta1, theta2]
        self.thresholds = [thrsh1, thrsh2]
        # network = {"thetas": thetas, "thresholds": thresholds, "loss": 0, "best": "False"}
        # return network
    
    def feed_forward(self, x):
        thetas = self.thetas
        for i in range(len(thetas)):
            if i == 0:
                z = np.dot(x.transpose(), thetas[i])
            else:
                z = np.dot(z, thetas[i])

            self.apply_threshold(z, i)
        return z

    @staticmethod
    def apply_threshold(z, i):
        if i != 1:
            for i in range(len(z)):
                if z[i] > 0:
                    z[i] = 1
                else:
                    z[i] = 0

    @staticmethod
    def original_loss(prediction, sample, goal_is_and=False):
        if sample["int_label"] == 3: label = 2
        elif sample["int_label"] == 1 or sample["int_label"] == 2: label = 1
        else: label = 0

        return int((prediction - label)**2)


def evaluate_q(population: list[NeuralNetwork], normalize, graph=NetworkGraph):
    population_q = []
    for network in tqdm(population, desc="Computing modularity for networks in population"):
        network.q = 0
        ng = graph(network)
        ng.convert2graph()
        ng.get_data()
        qvalue = compute_modularity(ng)
        if normalize:
            qvalue = normalize_q(qvalue)
        network.q = qvalue
        population_q.append(network.q)
    return population_q


# Non class functions
def evaluate_population(population: list[NeuralNetwork], samples, goal_is_and):
    population_loss = []
    pop_wrong = []
    pop_wrong_both = []
    pop_wrong_left = []
    pop_wrong_right = []
    pop_wrong_none = []
    # best_connection_loss = 1
    # best_performance_loss = 1
    for network in population:
        network.performance_loss = 0
        network.best = "false"
        network.both_count = 0
        network.left_count = 0
        network.right_count = 0
        network.none_count = 0

        for sample in samples:
            network.evaluate_network(sample, goal_is_and)

        # network.calculate_connection_loss()
        # network.normalize_losses(num_samples=len(samples), max_connections=106)
        network.combine_losses()
        population_loss.append(network.loss)
        pop_wrong_both.append(network.both_count)
        pop_wrong_left.append(network.left_count)
        pop_wrong_right.append(network.right_count)
        pop_wrong_none.append(network.none_count)

        # if network.performance_loss < best_performance_loss:
        #     best_performance_loss = network.performance_loss
        # if network.connection_loss < best_connection_loss:
        #     best_connection_loss = network.connection_loss
    pop_wrong = [pop_wrong_both, pop_wrong_left, pop_wrong_right, pop_wrong_none]
    return population_loss, pop_wrong#, best_performance_loss, best_connection_loss


def generate_population(n):
    population = []
    for i in tqdm(range(n), desc="Generating population"):
        network = NeuralNetwork()
        population.append(network)
    return population


def luc_generate_population(n):
    population = []
    for i in tqdm(range(n), desc="Generating Luc population"):
        network = LucNeuralNetwork()
        population.append(network)
    return population
