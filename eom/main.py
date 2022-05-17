import argparse
import copy
import itertools
import json
from math import ceil
import os
os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'
from os.path import join
from os import makedirs
import random
import shutil
import cProfile
import pstats
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from numba import njit
from tqdm import tqdm
import yaml

from graph import visualize_graph_data, NetworkGraph
from modularity import compute_modularity, normalize_q


def main(config):
    runname = config["runname"]
    checkpoint = config["checkpoint"]
    elite = config["elite"]
    gen_size = config["gen_size"]
    generations = config["generations"]
    mvg = config["mvg"]
    goal_is_and = config["goal_is_and"]
    qvalues = config["qvalues"]
    qvalue_interval = config["qvalue_interval"]
    mvg_frequency = config["mvg_frequency"]
    parents_perc = config["parents_perc"]
    early_stopping = config["early_stopping"]
    load_initial_gen = config["load_initial_gen"]
    initial_gen_path = config["initial_gen_path"]
    num_parents = int(gen_size * parents_perc)

    all_losses = []
    best_losses = []
    average_losses = []
    all_q = []
    best_q = []
    average_q = []
    parents_q = []
    counter = 0
    save_mode = False
    checking = True

    if load_initial_gen:
        population = load_weights(initial_gen_path)
    else:
        population = [build_network() for i in range(gen_size)]

    samples = generate_samples()

    clear_dirs(runname)
    save_weights(population[:10], runname, 0)
    visualize_graph_data(population[:10], runname, 0)

    for i in range(generations):
        print(f"\n ---- Run {runname}. Starting Gen {i}")

        # Varying the Loss Function
        if mvg and i % mvg_frequency == 0 and i != 0:
            goal_is_and = not goal_is_and
            print(f"Goal changed to goal_is_and={goal_is_and}")
        if goal_is_and: print(f"Goal is L AND R")
        else: print("Goal is L OR R")

        if qvalues and i % qvalue_interval == 0:
            population_q = evaluate_q(population, normalize=True)
            record_q(population_q, all_q, best_q, average_q, parents_q, int(gen_size*parents_perc))
            plot_q(best_q, average_q, parents_q, runname)

        if i > 0:
            # Main genetic algorithm code
            parents = select_best_score(population, all_losses[i - 1], num_parents)
            offspring = crossover(parents, gen_size, elite, parents_perc)
            population = mutate(offspring)
            if elite:
                population = parents + population

            if i % 50:
                plot_loss(best_losses, average_losses, runname)

            # Checkpoint
            if i % checkpoint == 0:
                visualize_graph_data(parents[:10], runname, i)
                save_weights(population[:10], runname, i)

        population_loss = evaluate_population(population, samples, goal_is_and, loss="loss", activation="tanh")
        record_loss(population_loss, all_losses, best_losses, average_losses)
        print("Loss: ", best_losses[i])

        if early_stopping:
            if best_losses[i] == 0:
                counter += 1
            else: counter = 0

            if counter > 50:
                print("Early Stop!")
                break

        # if checking and (counter > 30 or i > 2000):
        #     save_mode = True
        #     start, stop = i, i + 101
        #     print("Save mode active until generation ", stop)
        #     checking = False
        #
        # if save_mode:
        #     if start <= i <= stop:
        #         visualize_graph_data(population, runname, i)
        #         save_weights(population, runname, i)

    # Final operations
    plot_loss(best_losses, average_losses, runname)
    visualize_graph_data(population, runname, i)
    save_weights(parents[:10], runname, i)


def unit_test_feedforward():
    gen_size = 1000
    generations = 10
    population = [build_network() for i in range(gen_size)]
    samples = generate_samples()

    for i in range(generations):
        print(f"Gen {i}")
        population_loss = evaluate_population(population, samples, goal_is_and=True, activation="tanh")


# Generate 256 Samples
def and_label_sample(sample):
    left = False
    right = False

    if (sum(sample[0:4]) >= 3) or (sum(sample[0:2]) >= 1 and sum(sample[2:4]) == 0): left = True
    if (sum(sample[4:8]) >= 3) or (sum(sample[6:8]) >= 1 and sum(sample[4:6]) == 0): right = True

    if left and right: return 1
    elif left: return 0
    elif right: return 0
    else: return 0


def or_label_sample(sample):
    left = False
    right = False

    if (sum(sample[0:4]) >= 3) or (sum(sample[0:2]) >= 1 and sum(sample[2:4]) == 0): left = True
    if (sum(sample[4:8]) >= 3) or (sum(sample[6:8]) >= 1 and sum(sample[4:6]) == 0): right = True

    if left and right: return 1
    elif left: return 1
    elif right: return 1
    else: return 0


def generate_samples():
    """Much more eloquent way of doing things
    https://stackoverflow.com/questions/4928297/all-permutations-of-a-binary-sequence-x-bits-long"""
    samples = [dict({"pixels": np.array(seq), "and_label": and_label_sample(seq), "or_label": or_label_sample(seq)}) for seq in itertools.product([0,1], repeat=8)]
    # samples = [dict({"pixels": np.array(seq).reshape((1,len(seq))), "label": label_sample(seq)}) for seq in itertools.product([0,1], repeat=8)]
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
    network = {"thetas": thetas, "biases": biases, "loss": 0, "q": "n/a"}
    # apply_neuron_constraints(network)
    return network


def apply_neuron_constraints(network):
    # thetas = network["thetas"]
    for theta in network["thetas"]:
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

# Evaluate Network
"""Maybe if we can represent the networks in some abstract way and 
   then just use a regular ML lib to do the eval"""
def evaluate_population(population, samples, goal_is_and, loss, activation="tanh"):
    population_loss = []
    for i, network in enumerate(population):
        network[loss] = 0
        for sample in samples:
            evaluate_network(network, sample, loss, goal_is_and, activation)
        population_loss.append(network[loss])
    return population_loss


def evaluate_network(network, sample, loss, goal_is_and, activation="tanh"):
    x = sample["pixels"]
    prediction = feed_forward(network, x, activation)
    if goal_is_and:
        loss_ = calculate_loss_and(prediction, sample)
    else:
        loss_ = calculate_loss_or(prediction, sample)
    network[loss] += loss_


def feed_forward(network, x, activation="vanilla"):
    for i in range(len(network["thetas"])):
        if i == 0:
            # z = dot_py(x, network["thetas"][i])
            z = np.dot(x.transpose(), network["thetas"][i])
        else:
            # z = dot_py(z, network["thetas"][i])
            z = np.dot(z, network["thetas"][i])

        if activation == "vanilla":
            apply_threshold(z, network["biases"][i])
        if activation == "tanh":
            z = tanh_activation(z, network["biases"][i])
            # z2 = tanh_activation(z2, network["biases"][i])

    if (activation == "vanilla" and z > 0) or (activation == "tanh" and z >= 0):
        z = 1
    else: z = 0

    return z


@njit(fastmath=True)
def dot_py(A,B):
    m, n = A.shape
    p = B.shape[1]

    C = np.zeros((m,p))

    for i in range(0,m):
        for j in range(0,p):
            for k in range(0,n):
                C[i,j] += A[i,k]*B[k,j]
    return C


@njit(fastmath=True)
def tanh_activation(z, b):
    # a = np.sum([z.reshape(len(z),1), b], axis=0)
    a = z + b.reshape(len(b))
    return np.tanh(20*a)


def apply_threshold(z, t):
    for i in range(len(z)):
        if z[i] > t[i]:
            z[i] = 1
        else:
            z[i] = 0


def calculate_loss_and(prediction, sample):
    return int((prediction - sample["and_label"]) ** 2)


def calculate_loss_or(prediction, sample):
    return int((prediction - sample["or_label"]) ** 2)


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


def record_q(population_q, all_q, best_q, average_q, parents_q, num_parents):
    all_q.append(population_q)
    best_q.append(round(max(population_q), 3))
    average_q.append(round(sum(population_q) / len(population_q), 3))
    parents_q.append(round(sum(population_q[:num_parents]) / num_parents, 3))


def plot_q(best_scores, average_scores, parent_scores, runname):
    matplotlib.use("Agg")
    fig = plt.figure(figsize=(24,8))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)
    ax1.plot(best_scores, label='best Q')
    ax2.plot(average_scores, label='average Q')
    ax3.plot(parent_scores, label='Parents Average Q')
    ax1.set_xlabel('Generation (n)')
    ax1.set_ylabel('Q')
    ax1.set_title('Best Q Each Generation')
    ax1.legend()
    ax2.set_xlabel('Generation (n)')
    ax2.set_ylabel('Q')
    ax2.set_title('Average Q Each Generation')
    ax2.legend()
    ax3.set_xlabel('Generation (n)')
    ax3.set_ylabel('Q')
    ax3.set_title('Parents Average Q Each Generation')
    ax3.legend()

    file_path = join('Q_curves', f'Q_{runname}').replace("\\", "/")
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
        # child["parent1"] = parent_1["id"]
        # child["parent2"] = parent_2["id"]

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
def mutate(population, broadness=False, num_nodes=15):
    """Instead of iterating through each weight, just do the random pull the appropriate number of times,
       choose a number randomly from the appropriate domain, and then alter that node/weight/bias"""
    for i in range(len(population)):

        num_active_connections = count_connections(population[i])

        # Add connection
        if random.uniform(0,1) <= 0.2 and num_active_connections < 106:
            add_connection(population[i])

        # Remove connection
        if random.uniform(0, 1) <= 0.2 and num_active_connections > 0:
            remove_connection(population[i])

        # Mutate Threshold
        for t in range(num_nodes):
            if random.uniform(0,1) <= (1/24):
                mutate_threshold(population[i], t)

        # Mutate connection
        if broadness: prob_connection_mutate = .025
        elif not broadness: prob_connection_mutate = (2 / num_active_connections)
        if num_active_connections > 0:
            for theta in range(len(population[i]["thetas"])):
                for neuron in range(len(population[i]["thetas"][theta])):
                    for connection in range(len(population[i]["thetas"][theta][neuron])):
                        if population[i]["thetas"][theta][neuron][connection] != 0:
                            if random.uniform(0, 1) <= prob_connection_mutate:
                                mutate_connection(population[i], theta, neuron, connection)

        # apply_neuron_constraints(population[i])

    return population


def count_connections(network):
    count = 0
    for layer in network["thetas"]:
        count += np.count_nonzero(layer)
    return count


def add_connection(network):
    l1 = np.random.choice(4)
    while not check_for_value(network, l1, remove=False):
        l1 = np.random.choice(4)
    n1 = np.random.choice(network["thetas"][l1].shape[0])
    n2 = np.random.choice(network["thetas"][l1].shape[1])
    while network["thetas"][l1][n1][n2] != 0:
        # todo figure out way of avoiding long random loops. predefine what connections are 0
        n1 = np.random.choice(network["thetas"][l1].shape[0])
        n2 = np.random.choice(network["thetas"][l1].shape[1])
    network["thetas"][l1][n1][n2] = np.random.choice([-2, -1, 1, 2])


def remove_connection(network):
    l1 = np.random.choice(4)
    while not check_for_value(network, l1, remove=True):
        l1 = np.random.choice(4)
    n1 = np.random.choice(network["thetas"][l1].shape[0])
    n2 = np.random.choice(network["thetas"][l1].shape[1])
    while network["thetas"][l1][n1][n2] == 0:
        # todo figure out way of avoiding long random loops. predefine what connections are not 0
        n1 = np.random.choice(network["thetas"][l1].shape[0])
        n2 = np.random.choice(network["thetas"][l1].shape[1])
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


def mutate_connection_oldway(network, weight_num):
    index = map_weight_network(weight_num)
    weight_value = network["thetas"][index[0]][index[1]][index[2]]
    if weight_value != 0:
        change = np.random.choice([-1, 1])
        new_value = weight_value + change
        if -2 <= new_value <= 2:
            network["thetas"][index[0]][index[1]][index[2]] = new_value


def mutate_connection(network, theta, neuron, connection):
    weight_value = network["thetas"][theta][neuron][connection]
    if weight_value != 0:
        change = np.random.choice([-1, 1])
        new_value = weight_value + change
        if -2 <= new_value <= 2:
            if new_value == 0:
                new_value = weight_value + (2 * change)
            network["thetas"][theta][neuron][connection] = new_value


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
def select_best_score(population, scores, num_parents, reverse=False):
    # Sorts the population by loss scores from lowest to highest and returns the best
    sort = sorted(range(len(scores)), key=lambda k: scores[k], reverse=reverse)
    selected = [population[i] for i in sort[0:num_parents]]
    return selected


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


def save_weights(population, runname, gen):
    makedirs(f"saved_weights/{runname}/gen_{gen}", exist_ok=True)
    for i in range(len(population)):
        w_file = open(f"saved_weights/{runname}/gen_{gen}/network_{i}.json", "w")
        json.dump(population[i], w_file, default=default)
        # json.dump(population[i]["thetas"], w_file, default=default)
        # json.dump(population[i]["biases"], w_file, default=default)
        w_file.close()


def load_weights(weights_path):
    population = []
    for file in os.listdir(weights_path):
        w_file = open(f"{weights_path}/{file}", "r")
        network = json.load(w_file)
        the_hard_way(network)
        population.append(network)
        w_file.close()
    return population


def default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError('Not serializable')


def the_hard_way(network):
    keys = ["thetas", "biases"]
    for key in keys:
        for i in range(len(network[key])):
            for j in range(len(network[key][i])):
                if isinstance(network[key][i][j], list):
                    network[key][i][j] = np.array(network[key][i][j])
            if isinstance(network[key][i], list):
                network[key][i] = np.array(network[key][i])


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # Arguments
    parser = argparse.ArgumentParser(description='Evolutionary Algorithm for Networks.')
    parser.add_argument('--exp_id', type=str, help='Name of the experiment.')
    parser.add_argument('--trial_number', type=int, help='Specific iteration within the experiment.')
    args = parser.parse_args()
    exp_id = args.exp_id
    trial_num = args.trial_number

    # Load in the experiment configurations
    yaml_filename = "active_experiment.yaml"
    with open("config_files/"+yaml_filename, 'r') as file:
        config = yaml.safe_load(file)

    main(config, exp_id, trial_num)


    # makedirs(join('cprofile', runname).replace("\\", "/"), exist_ok=True)
    # samps = generate_samples()
    # right = sum([1 for samp in samps if samp["label"]=="right"])
    # left = sum([1 for samp in samps if samp["label"]=="left"])
    # both = sum([1 for samp in samps if samp["label"]=="both"])
    # none = sum([1 for samp in samps if samp["label"]=="none"])
    # x = right + left + both + none

    # for col in ["tottime"]:
    #     profiler = cProfile.Profile()
    #     profiler.enable()
    #     main(runname)
        # unit_test_feedforward()
        # profiler.disable()
        # with open(f'cprofile/{runname}/{col}.txt', 'w') as f:
        #     stats = pstats.Stats(profiler, stream=f)
        #     stats.strip_dirs()
        #     stats.sort_stats(col)
        #     stats.print_stats()
