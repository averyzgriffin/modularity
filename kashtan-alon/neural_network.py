"""
Contains functions related to the neural networks themselves
E.g. building the neural networks, computing feed-forward, thresholding, loss-functions.
This is called by the main.py module
"""

import numpy as np
import random
from tqdm import tqdm


def apply_neuron_constraints(network):
    thetas = network["thetas"]
    for theta in thetas:
        theta = theta.transpose()
        for node_num in range(len(theta)):
            total = sum(abs(theta[node_num]))
            while total > 3:
                choice = random.randint(0,len(theta.transpose())-1)
                if theta[node_num][choice] > 0:
                    theta[node_num][choice] -= 1
                elif theta[node_num][choice] < 0:
                    theta[node_num][choice] += 1
                total = sum(abs(theta[node_num]))


def apply_threshold(z, t):
    for i in range(len(z)):
        if z[i] > t[i]: z[i] = 1
        else: z[i] = 0


def build_network():
    theta1 = np.random.choice([-2,2], (8,8))
    theta2 = np.random.choice([-2,2], (8,4))
    theta3 = np.random.choice([-2,2], (4,2))
    theta4 = np.random.choice([-2,2], (2,1))
    thrsh1 = np.random.randint(-4,3, (8,1))
    thrsh2 = np.random.randint(-4,3, (4,1))
    thrsh3 = np.random.randint(-4,3, (2,1))
    thrsh4 = np.random.randint(-2,1, (1,1))

    thetas = [theta1, theta2, theta3, theta4]
    thresholds = [thrsh1, thrsh2, thrsh3, thrsh4]

    network = {"thetas": thetas, "thresholds": thresholds, "loss": 0}

    return network


def calculate_loss(prediction, sample, goal_is_and):
    if goal_is_and:
        if sample["int_label"] == 3: label = 1
        else: label = 0
    elif not goal_is_and:
        if sample["int_label"] == 1 or sample["int_label"] == 2 or sample["int_label"] == 3: label = 1
        else: label = 0

    return int((prediction - label)**2)


def evaluate_network(network, sample, goal_is_and):
    x = sample["pixels"]
    prediction = feed_forward(x, network)
    loss = calculate_loss(prediction, sample, goal_is_and)
    network["loss"] += loss


def evaluate_population(population, samples, goal_is_and):
    population_loss = []
    for network in population:
        network["loss"] = 0
        for sample in samples:
            evaluate_network(network, sample, goal_is_and)
        population_loss.append(network["loss"])
    return population_loss

def evaluate_q(population):
    population_q = []
    for network in population:
        network["q"] = 0
        ng = NetworkGraph(network)
        ng.convert2graph()
        ng.get_data()
        network["q"] = compute_modularity(ng)
        population_q.append(network["q"])
    return population_q


def feed_forward(x, network):
    thetas = network["thetas"]
    ts = network["thresholds"]
    for i in range(len(thetas)):
        if i == 0:
            z = np.dot(x.transpose(), thetas[i])
        else:
            z = np.dot(z, thetas[i])
        apply_threshold(z, ts[i])
    return z


def generate_population(n):
    population = []
    for i in tqdm(range(n), desc="Generating population"):
        network = build_network()
        apply_neuron_constraints(network)
        population.append(network)
    return population
