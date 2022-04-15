import json
import numpy as np
import os
import itertools

from graph import visualize_graph_data
from main import evaluate_population, select_best_loss, record_loss, crossover, mutate


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


def the_hard_way(network):
    keys = ["thetas", "biases"]
    for key in keys:
        for i in range(len(network[key])):
            for j in range(len(network[key][i])):
                if isinstance(network[key][i][j], list):
                    network[key][i][j] = np.array(network[key][i][j])
            if isinstance(network[key][i], list):
                network[key][i] = np.array(network[key][i])


samples = generate_samples()
population = []
all_losses, best_losses, average_losses = [],[],[]

# total = 0
# total2 = 0
# for k in range(221,240):
folder = f"saved_weights/mvg_gen1000_BasicSelection_addedconnectionCount_Qvalues/gen_221"
for file in os.listdir(folder):
    w_file = open(f"{folder}/{file}", "r")
    network = json.load(w_file)
    the_hard_way(network)
    population.append(network)
    w_file.close()

for k in range(10):
    print("\nGenereation ", k)

    # visualize_graph_data(population, "loaded_weights", k)

    or_loss = evaluate_population(population, samples, goal_is_and=False, loss="or_loss", activation="tanh")
    # and_loss = evaluate_population(population, samples, goal_is_and=True, loss="and_loss", activation="tanh")
    record_loss(or_loss, all_losses, best_losses, average_losses)
    count_or = sum([1 for i in or_loss if i == 0])
    # count_and = sum([1 for i in and_loss if i == 0])
    # total += count
    # total2 += count2

    if k > 0:
        viz = []
        for network in population:
            if network["or_loss"] == 0:
                viz.append(network)
                try:
                    viz.append(population[network["parent1"]])
                    viz.append(population[network["parent2"]])
                except Exception as e:
                    print(e)

        visualize_graph_data(viz, "loaded_weights", k+221)

    # Label networks
    sort = sorted(range(len(or_loss)), key=lambda k: or_loss[k])
    population = [population[i] for i in sort]
    for i in range(len(population)):
        population[i]["id"] = i

    # Main genetic algorithm code
    parents = population[:400] # select_best_loss(population, all_losses[k], 400)
    offspring = crossover(parents, 1000, True, .40)
    population = mutate(offspring)
    population = parents + population


    # sorted_parents = select_best_loss(population, loss, 1000)
    # print("OR Loss: ",   loss)
    # print("And Loss: ",  loss2)
    print("OR 0 Count: ",  count_or)
    # print("And 0 Count: ", count_and)

# print("Total OR 0 Count: ", total)
# print("Total AND 0 Count: ", total2)
# print("Average OR 0 Count: ", total  / 18)
# print("Average AND 0 Count: ", total2/ 18)

# loss1 = evaluate_population([population[0]], samples, goal_is_and=True, activation="tanh")
# loss2 = evaluate_population([population[1]], samples, goal_is_and=True, activation="tanh")
# loss3 = evaluate_population([population[0]], samples, goal_is_and=False, activation="tanh")
# loss4 = evaluate_population([population[1]], samples, goal_is_and=False, activation="tanh")
# print("Loss2: ", loss2)
# print("Loss3: ", loss3)
# print("Loss4: ", loss4)

check = 4