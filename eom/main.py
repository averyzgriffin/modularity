import copy
import itertools
from math import ceil
import os
from os.path import join
import random
import shutil

import io
import cProfile
import pstats
import matplotlib
from matplotlib import pyplot as plt
import numpy as np

from graph import visualize_graph_data, NetworkGraph


def main(runname):
    checkpoint = 10
    elite = False
    gen_size = 1000
    generations = 15
    mvg = False
    goal_is_and = True
    mvg_frequency = 20
    parents_perc = .40
    num_parents = int(gen_size * parents_perc)

    population = [build_network() for i in range(gen_size)]
    samples = generate_samples()
    all_losses = []
    best_losses = []
    average_losses = []

    for i in range(generations):
        print(f"\n Starting Gen {i}")

        # Varying the Loss Function
        if mvg and i % mvg_frequency == 0 and i != 0:
            goal_is_and = not goal_is_and
            print(f"Goal changed to goal_is_and={goal_is_and}")
        if goal_is_and: print(f"Goal is L AND R")
        else: print("Goal is L OR R")


        if i > 0:
            # Main genetic algorithm code
            parents = select_best_loss(population, all_losses[i-1], num_parents)
            offspring = crossover(parents, gen_size, elite, parents_perc)
            population = mutate(offspring)
            # if elite:
            #     population = parents + population

            # Checkpoint
            if i % checkpoint == 0:
                visualize_graph_data(parents[:10], runname, i)
                plot_loss(best_losses, average_losses, runname)

        population_loss = evaluate_population(population, samples, goal_is_and)
        record_loss(population_loss, all_losses, best_losses, average_losses)
        print("Loss: ", best_losses[i])

    # Final operations
    plot_loss(best_losses, average_losses, runname)
    visualize_graph_data(parents[:10], runname, i)


# Generate 256 Samples
def label_sample(sample):
    left = False
    right = False
    if sum(sample[0:4]) >= 1 or sum(sample[0:2]) >= 1: left = True
    if sum(sample[4:8]) >= 1 or sum(sample[6:8]) >= 1: right = True

    if left and right: return "both"
    elif left: return "left"
    elif right: return "right"
    else: return "neither"


def generate_samples():
    """Much more eloquent way of doing things
    https://stackoverflow.com/questions/4928297/all-permutations-of-a-binary-sequence-x-bits-long"""
    samples = [dict({"pixels": np.array(seq), "label": label_sample(seq)}) for seq in itertools.product([0,1], repeat=8)]
    return samples


# Build Network
def build_network():
    """I am curious if there is some way to represent these in a more optimized manner. e.g. using a ML library"""
    thetas = []
    biases = []
    input_size  = [8,8,4,2]
    output_size = [8,4,2,1]
    for i in range(4):
        thetas.append(np.random.randint(-2, 2, (input_size[i], output_size[i])))
        biases.append(np.random.randint(-4, 3, (output_size[i], 1)))
    biases[-1] = np.random.randint(-2, 1, (1,1))
    # apply_neuron_constraints()
    return {"thetas": thetas, "biases": biases, "loss": 0}


# Evaluate Network
"""Maybe if we can represent the networks in some abstract way and 
   then just use a regular ML lib to do the eval"""
def evaluate_population(population, samples, goal_is_and):
    population_loss = []
    for network in population:
        network["loss"] = 0
        for sample in samples:
            evaluate_network(network, sample, goal_is_and)
        population_loss.append(network["loss"])
    return population_loss


def evaluate_network(network, sample, goal_is_and):
    x = sample["pixels"]
    prediction = feed_forward(network, x)
    loss = calculate_loss(prediction, sample, goal_is_and)
    network["loss"] += loss


def feed_forward(network, x):
    for i in range(len(network["thetas"])):
        if i == 0:
            z = np.dot(x.transpose(), network["thetas"][i])
        else:
            z = np.dot(z, network["thetas"][i])

        # if i != len(network["thetas"]) - 1:
        # todo may want to do this differently. could be making a big deal
        apply_threshold(z, network["biases"][i])
    return z


def apply_threshold(z, t):
    for i in range(len(z)):
        if z[i] > t[i]:
            z[i] = 1
        else:
            z[i] = 0


def calculate_loss(prediction, sample, goal_is_and):
    if goal_is_and:
        if sample["label"] == "both":
            label = 1
        else:
            label = 0
    elif not goal_is_and:
        if sample["label"] == "left" or sample["label"] == "right" or sample["label"] == "both":
            label = 1
        else:
            label = 0

    return int((prediction - label) ** 2)


def record_loss(population_loss, all_losses, best_losses, average_losses):
    all_losses.append(population_loss)
    average_loss = round(sum(population_loss) / len(population_loss), 3)
    best_loss = round(min(population_loss), 3)
    best_losses.append(best_loss)
    average_losses.append(average_loss)


def plot_loss(best_scores, average_scores, runname):
    matplotlib.use("Agg")
    fig = plt.figure(figsize=(24,8))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.plot(best_scores, label='best loss')
    ax2.plot(average_scores, label='average loss')
    ax1.set_xlabel('Generation (n)')
    ax1.set_ylabel('Loss')
    ax1.set_title('Best Loss Each Generation')
    ax1.legend()
    ax2.set_xlabel('Generation (n)')
    ax2.set_ylabel('Loss')
    ax2.set_title('Average Loss Each Generation')
    ax2.legend()

    # plt.show()
    file_path = join('loss_curves', f'loss_{runname}').replace("\\", "/")
    plt.savefig(file_path+".png")

    fig.clear()
    plt.close(fig)


# Crossover Networks
def crossover(parents, gen_size, elite, parents_perc):
    new_gen = []

    if elite: num_child = int(gen_size*(1-parents_perc))
    else: num_child = int(gen_size)

    for i in range(num_child):
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
                    child["biases"][l][n] = selected_parents[choice]["biases"][l][n]

        new_gen.append(child)
    return new_gen


# Mutate Network
def mutate(population, num_nodes=15):
    """Instead of iterating through each weight, just do the random pull the appropriate number of times,
       choose a number randomly from the appropriate domain, and then alter that node/weight/bias"""
    num_active_connections = 100
    for i in range(len(population)):
        # Add connection
        if random.uniform(0,1) <= 0.2:
            add_connection(population[i])

        # Remove connection
        if random.uniform(0, 1) <= 0.2:
            remove_connection(population[i])

        # Mutate Threshold
        for t in range(num_nodes):
            if random.uniform(0,1) <= (1/24):
                mutate_threshold(population[i], t)

        # Mutate connection
        for w in range(num_active_connections):
            if random.uniform(0, 1) <= (2/num_active_connections):
                mutate_connection(population[i], w)

    return population


def add_connection(network):
    l1 = np.random.choice(3)
    while not check_for_value(network, l1, remove=False):
        l1 = np.random.choice(3)
    n1 = np.random.choice(len(network["thetas"][l1]))
    n2 = np.random.choice(len(network["thetas"][l1+1]))
    while network["thetas"][l1][n1][n2] != 0:
        # todo figure out way of avoiding long random loops. predefine what connections are 0
        n1 = np.random.choice(len(network["thetas"][l1]))
        n2 = np.random.choice(len(network["thetas"][l1+1]))
    network["thetas"][l1][n1][n2] = np.random.choice([-2, -1, 1, 2])


def remove_connection(network):
    l1 = np.random.choice(3)
    while not check_for_value(network, l1, remove=True):
        l1 = np.random.choice(3)
    n1 = np.random.choice(len(network["thetas"][l1]))
    n2 = np.random.choice(len(network["thetas"][l1+1]))
    while network["thetas"][l1][n1][n2] == 0:
        # todo figure out way of avoiding long random loops. predefine what connections are not 0
        n1 = np.random.choice(len(network["thetas"][l1]))
        n2 = np.random.choice(len(network["thetas"][l1+1]))
    network["thetas"][l1][n1][n2] = 0


def check_for_value(network, l1, remove=False):
    for w in network["thetas"][l1].flatten():
        if not remove and w == 0: return True
        elif remove and w != 0: return True


def mutate_threshold(network, node_num):
    index = map_node_network(node_num)
    change = np.random.choice([-1,1])
    bias_value = network["biases"][index[0]][index[1]]
    if -2 <= bias_value + change <= 2:
        network["biases"][index[0]][index[1]] += change


def mutate_connection(network, weight_num):
    index = map_weight_network(weight_num)
    weight_value = network["thetas"][index[0]][index[1]][index[2]]
    if weight_value != 0:
        change = np.random.choice([-1, 1])
        if -2 <= weight_value + change <= 2:
            network["thetas"][index[0]][index[1]][index[2]] += change


def map_node_network(node_num):
    cutoffs = [8, 4, 2, 1]
    for c in range(len(cutoffs)):
        if node_num > cutoffs[c]-1:
            node_num = node_num - cutoffs[c]
        else:
            return (c, node_num)


def map_weight_network(weight_num):
    cutoffs1 = [64, 32, 8, 2]
    cutoffs2 = [8, 4, 2, 1]
    for l in range(len(cutoffs1)):
        if weight_num > cutoffs1[l] - 1:
            weight_num = weight_num - cutoffs1[l]
        else:
            s = int(ceil((weight_num+1) / cutoffs2[l]))
            t = (weight_num+1) - ((s-1)*cutoffs2[l])
            index = (l, s-1, t-1)
            return index


# Multiobjective Selection
def select_best_loss(population, scores, num_parents):
    # Sorts the population by loss scores from lowest to highest and returns the best
    sort = sorted(range(len(scores)), key=lambda k: scores[k])
    selected = [population[i] for i in sort[0:num_parents]]
    return selected


def clear_dirs(runname):
    folders = ['graphviz_plots', 'networkx_graphs', 'saved_weights']
    for folder in folders:
        dir_path = join(folder, runname).replace("\\", "/")
        if os.path.exists(dir_path):
            for filename in os.listdir(dir_path):
                file_path = os.path.join(dir_path, filename).replace("\\", "/")
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    runname = "2013profiling" # "2013paper_000"
    clear_dirs(runname)

    for col in ["ncalls","tottime","cumtime"]:
        profiler = cProfile.Profile()
        profiler.enable()
        main(runname)
        profiler.disable()
        with open(f'cprofile/{col}.txt', 'w') as f:
            stats = pstats.Stats(profiler, stream=f)
            stats.strip_dirs()
            stats.sort_stats(col)
            stats.print_stats()
