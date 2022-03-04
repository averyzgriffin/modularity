"""
Run this module to get the max Q value used when normalizing Q
"""

import cProfile
import matplotlib
import yaml

from data_save import save_weights, save_q
from data_viz import plot_q, record_q, visualize_networks
from genetic_algo import crossover, mutate, select_best_qvalue
from neural_network import generate_population, evaluate_q
from network_graphs import convert_networks


def main(population, generations, p_m, checkpoint, runname, elite):
    num_parents = int(len(population)*.2)

    all_q = []
    best_q = []
    average_q = []
    matplotlib.use("Agg")

    visualize_networks(population[:10], runname, 0)
    convert_networks(population[:10], runname, 0)
    # save_weights(population[:10], runname, 0)

    for i in range(generations):
        print(f"\n ---- Run {runname}. Starting Gen {i}")
        print("mutation rate: ", p_m)

        if i > 0:
            # Main genetic algorithm code
            parents = select_best_qvalue(population, all_q[i-1], num_parents)
            offspring = crossover(parents, gen_size, elite)
            population = mutate(offspring, p_m)
            if elite:
                population = parents + population

            # Save experiment data at every checkpoint
            if i % checkpoint == 0:
                convert_networks(parents[:10], runname, i)
                plot_q(best_q, average_q, runname)
                save_weights(parents[:10], runname, i)
                visualize_networks(parents[:10], runname, i)

        population_q = evaluate_q(population, normalize=False)
        record_q(population_q, all_q, best_q, average_q)

    # Save experiment data at the very end
    convert_networks(parents[:10], runname, generations)
    plot_q(best_q, average_q, runname)
    save_weights(parents[:10], runname, generations)
    visualize_networks(parents[:10], runname, generations)
    save_q(best_q[-1], runname)


if __name__ == "__main__":

    # Load in the experiment configurations
    with open("experiment.yaml", 'r') as file:
        config = yaml.safe_load(file)

    num_samples = config["num_samples"]
    gen_sizes = config["gen_sizes"]
    mutation_rates = config["mutation_rates"]
    generations = config["generations"]
    checkpoint = config["checkpoint"]
    elites = config["elite"]

    # Main loop for running experiment. Loops through hyperparamters
    for gen_size in gen_sizes:
        for elite in elites:
            for p_m in mutation_rates:
                if config["runname"]: runname = config["runname"]
                else: runname = f"maxQ_elite{elite}_gensize{gen_size}_pm{p_m}"
                gen_0 = generate_population(gen_size)
                # cProfile.run('main(gen_0, generations, p_m, checkpoint, runname, elite)')
                main(gen_0, generations, p_m, checkpoint, runname, elite)










