"""
Contains all function related to the genetic algorithm itself.
E.g. selection, crossover, and mutation.
This is called by the main.py module
"""

import copy
import numpy as np
import random

from neural_network import apply_neuron_constraints


def crossover(parents, gen_size):
    new_gen = []
    for i in range(int(gen_size*.80)):
        # Select parents
        parent_1 = parents[np.random.randint(0,len(parents))]
        parent_2 = parents[np.random.randint(0,len(parents))]
        selected_parents = [parent_1, parent_2]

        # Copy one parent. Will be basis for child (speeds up crossover)
        child = copy.deepcopy(parent_1)

        # Loop through each weight and activation of the child and randomly assign parent
        for l in range(len(child["thetas"])):
            for n in range(len(child["thetas"][l].transpose())):
                choice = random.choice([0,1])
                # If 1, then we swap in parent 2 genes. Else if 0, keep parent 1 genes
                if choice:
                    child["thetas"][l].transpose()[n] = selected_parents[choice]["thetas"][l].transpose()[n]
                    child["thresholds"][l][n] = selected_parents[choice]["thresholds"][l][n]

        new_gen.append(child)
    return new_gen


def mutate(networks, p_m):
    # Loop through each weight/threshold of each neuron of each layer of each network
    for i in range(len(networks)):
        for l in range(len(networks[i]["thetas"])):
            for n in range(len(networks[i]["thetas"][l].transpose())):
                for w in range(len(networks[i]["thetas"][l].transpose()[n])):
                    # Sample randomly. If less than the p_m, mutate the weight.
                    if random.uniform(0,1) < p_m:
                        networks[i]["thetas"][l].transpose()[n][w] = random.randint(-2, 2)
                # Sample randomly. If less than the p_m, mutate the threshold value.
                if random.uniform(0, 1) < p_m:
                    if l < 3: networks[i]["thresholds"][l][n] = random.randint(-4, 3)
                    else: networks[i]["thresholds"][l][n] = random.randint(-2, 1)

        apply_neuron_constraints(networks[i])

    return networks


def select_best(population, scores, num_parents):
    # Sorts the population by loss scores from lowest to highest and returns the best
    sort = sorted(range(len(scores)), key=lambda k: scores[k])
    selected = [population[i] for i in sort[0:num_parents]]
    return selected







