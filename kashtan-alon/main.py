"""
Run this module to run the actual neural-net evolution experiement
"""

import json
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import yaml

from data_save import save_weights, save_q
from data_viz import plot_loss, record_loss, visualize_networks, plot_q, record_q
from generate_labeled_data import load_samples, generate_samples
from genetic_algo import crossover, mutate, select_best_loss
from neural_network import evaluate_population, generate_population, evaluate_q
from network_graphs import convert_networks


def main(samples, population, generations, p_m, goal, checkpoint, runname, mvg_frequency, elite):
    matplotlib.use("Agg")

    goal_is_and = True
    num_parents = int(len(population)*.5)

    all_losses = []
    best_losses = []
    average_losses = []

    all_q = []
    best_q = []
    average_q = []

    visualize_networks(population[:10], runname, 0)
    convert_networks(population[:10], runname, 0)
    save_weights(population[:10], runname, 0)

    for i in range(generations):
        print(f"\n ---- Run {runname}. Starting Gen {i}")

        # Varying the Loss Function
        if goal == "mvg" and i % mvg_frequency == 0 and i != 0:
            goal_is_and = not goal_is_and
            print(f"Goal changed to goal_is_and={goal_is_and}")
        if goal_is_and: print(f"Goal is L AND R")
        else: print("Goal is L OR R")

        # # Varying the mutation rate
        # # if i % 500 == 0 and i != 0:
        # #     p_m /= 10
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
            offspring = crossover(parents, gen_size, elite)
            population = mutate(offspring, p_m)
            if elite:
                population = parents + population

            # Save experiment data at every checkpoint
            if i % checkpoint == 0:
                convert_networks(parents[:10], runname, i)
                plot_loss(best_losses, average_losses, runname)
                save_weights(parents[:10], runname, i)
                visualize_networks(parents[:10], runname, i)

                population_q = evaluate_q(population, normalize=True) # TODO factor in randQ and maxQ
                record_q(population_q, all_q, best_q, average_q)
                plot_q(best_q, average_q, runname)

        population_loss = evaluate_population(population, samples, goal_is_and)
        record_loss(population_loss, all_losses, best_losses, average_losses)

    # Save experiment data at the very end
    convert_networks(parents[:10], runname, generations)
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
    samples = generate_samples(num_samples)

    for gen_size in gen_sizes:
        for goal in goals:
            for elite in elites:
                for p_m in mutation_rates:
                    for mvg_frequency in mvg_frequencies:
                        if goal == "fixed": mvg_frequency = 0
                        if config["runname"]: runname = config["runname"]
                        else: runname = f"1no2_Qvalue_elite{elite}_goal{goal}_mvg{mvg_frequency}_gensize{gen_size}_pm{p_m}"
                        gen_0 = generate_population(gen_size)
                        main(samples, gen_0, generations, p_m, goal, checkpoint, runname, mvg_frequency, elite)










