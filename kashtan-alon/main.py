from generate_labeled_data import generate_samples
from build_networks import generate_population
from data import plot_results, record_loss, save_networks, setup_savedir, visualize_networks
from neural_network import evaluate_population
from genetic_algo import crossover, mutate, select_best
# from visualize_nets import

import time
import yaml


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
                plot_results(best_losses, average_losses, runname)

        population_loss = evaluate_population(population, samples, i, mvg)
        record_loss(population_loss, all_losses, best_losses, average_losses)

        # if i % checkpoint == 0 or i == generations-1:
        #     visualize_networks(parents, runname, i) # todo change to parents somehow

    # plot_results(best_losses, average_losses, runname)
    # save_networks(best_scores, average_scores, total_scores) todo


if __name__ == "__main__":

    with open("experiment.yaml", 'r') as file:
        config = yaml.safe_load(file)

    num_samples = config["num_samples"]
    gen_sizes = config["gen_sizes"]
    mutation_rates = config["mutation_rates"]
    generations = config["generations"]
    mvg = config["mvg"]
    checkpoint = config["checkpoint"]
    if config["runname"]:
        runname = config["runname"]
    else:
        runname = f"RUN_mvg{mvg}_gensize{gen_sizes}_mr{mutation_rates}_{time.time()}"

    samples = generate_samples(num_samples)
    for p_m in mutation_rates:
        for gen_size in gen_sizes:
            # runname = f"RUN_mvg_gensize{gen_size}_pm{p_m}"
            setup_savedir(runname)
            gen_0 = generate_population(gen_size)
            main(samples, gen_0, generations, mvg, checkpoint, runname)










