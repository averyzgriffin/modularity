"""
Run this module to inspect exactly how evolved networks change from goal to goal
"""

import json
import numpy as np
import yaml

from data_viz import plot_loss, record_loss, save_loss_to_csv, setup_savedir, visualize_networks
from generate_labeled_data import load_samples
from genetic_algo import crossover, mutate, select_best
from neural_network import evaluate_population, generate_population


def default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError('Not serializable')


def the_hard_way(network):
    keys = ["thetas", "thresholds"]
    for key in keys:
        for i in range(len(network[key])):
            for j in range(len(network[key][i])):
                if isinstance(network[key][i][j], list):
                    network[key][i][j] = np.array(network[key][i][j])
            if isinstance(network[key][i], list):
                network[key][i] = np.array(network[key][i])

def load_networks():
    population = []
    for i in range(100):
        w_file = open(f"saved_weights/network_{i}.json", "r")
        network = json.load(w_file)
        the_hard_way(network)
        population.append(network)
        w_file.close()
    return population


def main(samples, population, generations, mvg, checkpoint, runname, mvg_frequency):
    num_parents = int(len(population)*.2)

    all_losses = []
    best_losses = []
    average_losses = []

    # visualize_networks(population, runname, 0)

    for i in range(generations):
        print("\n ---- Starting Gen ", i)
        if i > 0:
            parents = select_best(population, all_losses[i-1], num_parents)
            offspring = crossover(parents, gen_size)
            population = mutate(offspring, p_m)

            if i % checkpoint == 0:
                visualize_networks(parents, runname, i)
                plot_loss(best_losses, average_losses, runname)

        population_loss = evaluate_population(population, samples, i, mvg, mvg_frequency)
        record_loss(population_loss, all_losses, best_losses, average_losses)
        print(best_losses)


    # for i in range(len(parents)):
    #     w_file = open(f"saved_weights/{runname}/network_{i}.json", "w")
    #     json.dump(parents[i], w_file, default=default)
    #     w_file.close()

    visualize_networks(parents, runname, generations)
    plot_loss(best_losses, average_losses, runname)


if __name__ == "__main__":

    # Load in the experiment configurations
    with open("experiment.yaml", 'r') as file:
        config = yaml.safe_load(file)

    num_samples = config["num_samples"]
    gen_sizes = config["gen_sizes"]
    mutation_rates = config["mutation_rates"]
    generations = config["generations"]
    mvg = config["mvg"]
    mvg_frequencies = config["mvg_frequency"]
    checkpoint = config["checkpoint"]

    # Main loop for running experiment. Loops through hyperparamters
    samples = load_samples(num_samples, "samples")
    gen_0 = load_networks()
    for p_m in mutation_rates:
        for gen_size in gen_sizes:
            for mvg_frequency in mvg_frequencies:
                if config["runname"]: runname = config["runname"]
                else: runname = f"mvg{mvg_frequency}_gensize{gen_size}_pm{p_m}"
                setup_savedir(runname)
                main(samples, gen_0, generations, mvg, checkpoint, runname, mvg_frequency)









