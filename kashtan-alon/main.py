"""
Run this module to run the actual neural-net evolution experiement
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

def main(samples, population, generations, p_m, goal, checkpoint, runname, mvg_frequency, elite):
    goal_is_and = True
    num_parents = int(len(population)*.2)

    all_losses = []
    best_losses = []
    average_losses = []

    visualize_networks(population[:10], runname, 0)

    for i in range(generations):
        print("\n ---- Starting Gen ", i)

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
            parents = select_best(population, all_losses[i-1], num_parents)
            offspring = crossover(parents, gen_size)
            population = mutate(offspring, p_m)
            population = parents + population

            if i % checkpoint == 0:
                visualize_networks(parents, runname, i)
                plot_loss(best_losses, average_losses, runname)

        population_loss = evaluate_population(population, samples, goal_is_and)
        record_loss(population_loss, all_losses, best_losses, average_losses)

    # Saved the weights at the very end
    for i in range(len(parents)):
        w_file = open(f"saved_weights/network_{i}.json", "w")
        json.dump(parents[i], w_file, default=default)
        w_file.close()

    # Save the loss score and graphs of the networks at the very end
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
    goals = config["goals"]
    mvg_frequencies = config["mvg_frequency"]
    checkpoint = config["checkpoint"]

    # Main loop for running experiment. Loops through hyperparamters
    samples = load_samples(num_samples, "samples")
    for gen_size in gen_sizes:
        for goal in goals:
            for elite in elites:
                for p_m in mutation_rates:
                    for mvg_frequency in mvg_frequencies:
                        if goal == "fixed": mvg_frequency = 0
                        if config["runname"]: runname = config["runname"]
                        else: runname = f"elite{elite}_goal{goal}_mvg{mvg_frequency}_gensize{gen_size}_pm{p_m}"
                        gen_0 = generate_population(gen_size)
                        main(samples, gen_0, generations, p_m, goal, checkpoint, runname, mvg_frequency, elite)










