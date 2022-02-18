import copy
import json
import numpy as np
import random

from build_networks import apply_neuron_constraints
from data import visualize_solo_network

def crossover(parents, gen_size):
    # If anything goes wrong, this function is complicated enough to warrant inspection
    new_gen = []
    for i in range(gen_size):
        parent_1 = parents[np.random.randint(0,len(parents))]
        parent_2 = parents[np.random.randint(0,len(parents))]
        selected_parents = [parent_1, parent_2]
        template = copy.deepcopy(parent_1) # it's crucial that we copy parent 1 for this function to work
        for l in range(len(template["thetas"])):
            for n in range(len(template["thetas"][l].transpose())):
                choice = random.choice([0,1])
                if choice:
                    # Swap in parent 2 genes
                    template["thetas"][l].transpose()[n] = selected_parents[choice]["thetas"][l].transpose()[n]
                    template["thresholds"][l][n] = selected_parents[choice]["thresholds"][l][n]
        new_gen.append(template)
    return new_gen


def default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError('Not serializable')


def mutate(networks, p_m):
    for i in range(len(networks)):
        for l in range(len(networks[i]["thetas"])):
            for n in range(len(networks[i]["thetas"][l].transpose())):
                for w in range(len(networks[i]["thetas"][l].transpose()[n])):
                    if random.uniform(0,1) < p_m:
                        new_value = random.randint(-2, 2) # todo verify this makes sense
                        networks[i]["thetas"][l].transpose()[n][w] = new_value
                if random.uniform(0, 1) < p_m:
                    if l < 3: networks[i]["thresholds"][l][n] = random.randint(-4, 3)
                    else: networks[i]["thresholds"][l][n] = random.randint(-2, 1)

        # w_file = open(f"solo_networks/preconstraint_{i}_.json", "w")
        # json.dump(networks[i], w_file, default=default)
        # w_file.close()
        # visualize_solo_network(networks[i], name=f"{i}_pre")

        apply_neuron_constraints(networks[i])

        # w_file = open(f"solo_networks/postconstraint_{i}_.json", "w")
        # json.dump(networks[i], w_file, default=default)
        # w_file.close()
        # visualize_solo_network(networks[i], name=f"{i}_post")

    return networks


def select_best(population, scores, num_parents):
    sort = sorted(range(len(scores)), key=lambda k: scores[k])
    selected = [population[i] for i in sort[0:num_parents]]
    return selected







