import numpy as np
import random

from data import visualize_solo_network

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


def generate_population(n):
    population = []
    for i in range(n):
        network = build_network()
        apply_neuron_constraints(network)
        population.append(network)
    return population

# population = generate_population(1)
# for network in population:
#     apply_neuron_constraints(network)
