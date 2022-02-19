"""
Run this module to run the actual neural-net evolution experiement
"""

import yaml

from data_viz import plot_loss, record_loss, save_loss_to_csv, setup_savedir, visualize_networks
from generate_labeled_data import load_samples
from genetic_algo import crossover, mutate, select_best
from neural_network import evaluate_population, generate_population


def main(samples, population, generations, mvg, checkpoint, runname):
    num_parents = int(len(population)*.2)

    all_losses = []
    best_losses = []
    average_losses = []

    visualize_networks(population, runname, 0)

    for i in range(generations):
        print("\n ---- Starting Gen ", i)
        if i > 0:
            parents = select_best(population, all_losses[i-1], num_parents)
            offspring = crossover(parents, gen_size)
            population = mutate(offspring, p_m)

            if i % checkpoint == 0 or i == generations - 1:
                visualize_networks(parents, runname, i)
                plot_loss(best_losses, average_losses, runname)

        population_loss = evaluate_population(population, samples, i, mvg)
        record_loss(population_loss, all_losses, best_losses, average_losses)


if __name__ == "__main__":

    # Load in the experiment configurations
    with open("experiment.yaml", 'r') as file:
        config = yaml.safe_load(file)

    num_samples = config["num_samples"]
    gen_sizes = config["gen_sizes"]
    mutation_rates = config["mutation_rates"]
    generations = config["generations"]
    mvg = config["mvg"]
    checkpoint = config["checkpoint"]

    # Main loop for running experiment. Loops through hyperparamters
    samples = load_samples(num_samples, "samples")
    for p_m in mutation_rates:
        for gen_size in gen_sizes:
            if config["runname"]: runname = config["runname"]
            else: runname = f"RUN_mvg{mvg}_gensize{gen_size}_pm{p_m}"
            setup_savedir(runname)
            gen_0 = generate_population(gen_size)
            main(samples, gen_0, generations, mvg, checkpoint, runname)










