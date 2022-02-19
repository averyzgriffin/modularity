import numpy as np
import yaml

from generate_labeled_data import generate_samples
from build_networks import generate_population


def feed_forward(x, network):
    thetas = network["thetas"]
    ts = network["thresholds"]
    for i in range(len(thetas)):
        if i == 0: z = np.dot(x.transpose(), thetas[i])
        else: z = np.dot(z, thetas[i])
        apply_threshold(z, ts[i])
    return z


def apply_threshold(z, t):
    for i in range(len(z)):
        if z[i] > t[i]: z[i] = 1
        else: z[i] = 0


def calculate_loss(prediction, sample, gen_num, mvg):
    goal_is_and = True
    if mvg and gen_num % 2 == 0:
        goal_is_and = not goal_is_and
    if goal_is_and:
        if sample["int_label"] == 3: label = 1
        else: label = 0
    elif not goal_is_and:
        if sample["int_label"] == 1 or sample["int_label"] == 2: label = 1
        else: label = 0

    return int((prediction - label)**2)


def evaluate_network(network, sample, gen_num, mvg):
    x = sample["pixels"]
    prediction = feed_forward(x, network)
    loss = calculate_loss(prediction, sample, gen_num, mvg)
    network["loss"] += loss


def evaluate_population(population, samples, gen_num, mvg):
    population_loss = []
    for network in population:
        network["loss"] = 0
        for sample in samples:
            evaluate_network(network, sample, gen_num, mvg)
        population_loss.append(network["loss"])
    return population_loss


# samples = generate_samples(100)
# population = generate_population(200)
# loss_scores = []
#
# for network in population:
#     for sample in samples:
#         evaluate_network(network, sample)
#     loss_scores.append(network["loss"])
#
# parents = choose_parents(population, loss_scores)
