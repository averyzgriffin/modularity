"""
Run this module to run the actual neural-net evolution experiement
"""

import json
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import yaml

from data_save import save_weights, save_q, clear_dirs
from data_viz import plot_loss, record_loss, visualize_networks, plot_q, record_q
from generate_labeled_data import generate_samples, load_samples, filter_samples, generate_simples_samples
from genetic_algo import crossover, mutate, select_best_loss
from neural_network import evaluate_population, evaluate_q, luc_generate_population
from network_graphs import visualize_graph_data


def main(samples, population, generations, p_m, goal, checkpoint, runname, mvg_frequency, elite):
    matplotlib.use("Agg")

    count = 0
    detected_change = False
    goal_is_and = True
    parents_perc = .30
    num_parents = int(len(population)*parents_perc)

    all_losses = []
    best_losses = []
    average_losses = []

    all_q = []
    best_q = []
    average_q = []

    visualize_networks(population[:20], runname, 0)
    visualize_graph_data(population[:20], runname, 0)
    # save_weights(population, runname, 0)

    for i in range(generations):
        print(f"\n ---- Run {runname}. Starting Gen {i}")

        # Varying the Loss Function
        if goal == "mvg" and i % mvg_frequency == 0 and i != 0:
            goal_is_and = not goal_is_and
            print(f"Goal changed to goal_is_and={goal_is_and}")
        if goal == "mvg":
            if goal_is_and: print(f"Goal is L AND R")
            else: print("Goal is L OR R")

        # # Varying the mutation rate
        # if i % 200 == 0 and i != 0:
        #     p_m /= 2
        # if i == 500:
        #     p_m = .01
        # if i == 1000:
        #     p_m = .001
        # if i == 5000:
        #     p_m = .0001
        print("mutation rate: ", p_m)

        if i > 0:
            # Main genetic algorithm code
            parents = select_best_loss(population, all_losses[i-1], num_parents)
            offspring = crossover(parents, gen_size, elite, parents_perc)
            population = mutate(offspring, p_m)
            if elite:
                population = parents + population

            # Stuff we only want happening every checkpoint. e.g. saving experiment data
            if i % checkpoint == 0:
                plot_loss(best_losses, average_losses, runname)
                # visualize_graph_data(parents, runname, i-1)
                # visualize_networks(parents, runname, i)
                # save_weights(parents[:10], runname, i)

                # Computing modularity metrics. Expensive operation.
                population_q = evaluate_q(population, normalize=True) # TODO factor in randQ and maxQ
                record_q(population_q, all_q, best_q, average_q)
                plot_q(best_q, average_q, runname)

        if detected_change:
            print("Detected change post")
            visualize_graph_data(parents, runname, i-1)
            detected_change = False

        # Compute loss each generation
        population_loss = evaluate_population(population, samples, goal_is_and)
        record_loss(population_loss, all_losses, best_losses, average_losses)
        print("Loss: ", best_losses[i])

        if i>0:
            if best_losses[i] != best_losses[i-1]:
                print("Detected change prior")
                visualize_graph_data(parents, runname, i - 1)
                detected_change = True

        if best_losses[i] == 0:
            count += 1

        if count > 10:
            break

    # Save experiment data at the very end
    visualize_graph_data(parents[:], runname, generations)
    plot_loss(best_losses, average_losses, runname)
    save_weights(parents[:10], runname, generations)
    visualize_networks(parents[:10], runname, generations)

    population_q = evaluate_q(population, normalize=True)
    record_q(population_q, all_q, best_q, average_q)
    plot_q(best_q, average_q, runname)
    save_q(best_q[-1], runname)


if __name__ == "__main__":

    # Load in the experiment configurations
    with open("experiment.yaml", 'r') as file:
        config = yaml.safe_load(file)

    num_samples = config["num_samples"]
    gen_sizes = config["gen_sizes"]
    mutation_rates = config["mutation_rates"]
    generations = config["generations"]
    goals = config["goals"]
    mvg_frequencies = config["mvg_frequency"]
    checkpoint = config["checkpoint"]
    elites = config["elite"]

    # Main loop for running experiment. Loops through hyperparamters
    # samples = load_samples(num_samples, "samples")
    # filtered_samples = filter_samples(luc_samples, [3, 1, 2])
    # luc_samples = generate_samples(61)
    luc_samples = generate_simples_samples()

    for gen_size in gen_sizes:
        for goal in goals:
            for elite in elites:
                for p_m in mutation_rates:
                    for mvg_frequency in mvg_frequencies:
                        if goal == "fixed": mvg_frequency = 0
                        if config["runname"]: runname = config["runname"]
                        else: runname = f"LucSimplified_SimpleSamples_gensize{gen_size}_pm{p_m}_001"

                        clear_dirs(runname)
                        gen_0 = luc_generate_population(gen_size)
                        main(luc_samples, gen_0, generations, p_m, goal, checkpoint, runname, mvg_frequency, elite)










