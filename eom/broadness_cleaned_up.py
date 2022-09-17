"""
Measures broadness
"""
from csv import writer
import copy
from os import path
import os

from main import build_network, generate_samples, mutate, load_weights, makedirs
import numpy as np
from tqdm import tqdm
import tqdm as td

# from graph import NetworkGraph
from main_p import evaluate_population, evaluate_q




def get_average(i, n):
    try:
        return round(i / (n), 3)
    except ZeroDivisionError:
        return 0


def main():

    rootdir = r"C:\Users\avery\PycharmProjects\modularity\eom\saved_weights\main_run"
    for root, dirs, files in os.walk(rootdir):
        if len(files) > 0:
            split = root.split("\\")
            if split[-1] == "gen_19999":
                trial_num = split[-2].split("_")[1]
                goal = split[-3].split("_")[-1]

                if goal == "FixedOR": goal_is_and = False
                elif goal == "FixedAND": goal_is_and = True
                elif goal == "MVG": goal_is_and = False # get_goal(19999)

                print("\n",goal)
                print(trial_num)

                broadness(root, goal_is_and, trial_num)


def broadness(path, goal, trial_num):
    population = load_weights(path)
    samples = generate_samples()

    # goal_is_and = get_goal()

    with td.trange(len(population)) as t:
        t.set_description(f"Goal {'AND' if goal else 'OR'}. Trial {trial_num}. Measuring Broadness on population")

        # for n in tqdm(range(len(population)), desc=f"Goal {'AND' if goal else 'OR'}. Trial {trial_num}. Measuring Broadness on population"):
        for n in t:
            network = population[n]
            loss = measure_loss(network, goal, samples)
            broadness, variance_losses = measure_broadness(network, loss, goal, samples)
            Qs = measure_modularity(network)
            t.set_postfix(Loss=loss, Broadness=broadness, Variance=variance_losses, QL=Qs[0], QGN=Qs[1])
            # save_csv(goal, trial_num, n, loss, broadness, variance_losses, Qs)

def measure_loss(network, goal, samples):
    return evaluate_population([network], samples, goal, loss="loss", activation="tanh")[0]


def measure_broadness(network, starting_loss, goal, samples):
    mutation_steps = 1
    simulations = 100
    losses_in_simulation = []

    for s in range(simulations):
        population = [copy.deepcopy(network)]

        for i in range(mutation_steps):
            population = mutate(population, broadness=True)
            loss = evaluate_population(population, samples, goal, loss="loss", activation="tanh")[0]
            losses_in_simulation.append(loss)

    average_loss = np.mean(losses_in_simulation)
    broadness = average_loss - starting_loss
    variance_loss = np.var(losses_in_simulation)

    return broadness, variance_loss


def measure_modularity(network):
    qL = evaluate_q(network, inhouse=False, normalize=False, method="louvain", absval=True, partition_weights=True, q_weights='weight')
    qG = evaluate_q(network, inhouse=False, normalize=False, method="gn", absval=True, partition_weights=True, q_weights='weight')
    return [qL, qG]


def get_goal(gen):
    i = 0
    while gen > 20:
        gen -= 20
        i += 1
    if i % 2 == 0:
        return True
    return False


def save_csv(goal, trial_num, model_num, starting_loss, qs: list, broadness, variance_losses):
    output_contents = [goal, trial_num, model_num, starting_loss, broadness, variance_losses] + qs
    column_labels = ["Goal", "Trial #", "Model #", "Starting Loss", "Broadness (Delta L 1-Step)", "Variance Loss 1 Step", "Q Lou", "Q GNW"]
    csv_file = f"measuring_broadness/comprehensive_measure_broadness.csv"

    mode = "a" if path.isfile(csv_file) else "w"
    with open(csv_file, mode, newline='') as write_obj:
        csv_writer = writer(write_obj)
        if mode == "w":
            csv_writer.writerow(column_labels)
        csv_writer.writerow(output_contents)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()







