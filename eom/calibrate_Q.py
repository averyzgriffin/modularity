"""
Run this module to get the max Q value used when normalizing Q
"""

from datetime import datetime
import cProfile
import matplotlib
import yaml

from main import plot_q, record_q, build_network, generate_samples, clear_dirs, select_best_score, mutate, crossover, evaluate_q
from graph import visualize_graph_data


def compute_Qmax(sim_num):
    date = datetime.now().strftime("%x").replace('/', '_')
    runname = f"Qmax_{date}"
    # if sim_num == 0:
    #     generations = 20000
    #     checkpoint = 1000
    # else:
    generations = 2000
    checkpoint = 250

    gen_size = 1000
    elite = True
    parents_perc = .40
    num_parents = int(gen_size * parents_perc)

    all_q = []
    best_q = []
    average_q = []
    parents_q = []

    population = [build_network() for i in range(gen_size)]

    clear_dirs(runname)
    visualize_graph_data(population[:10], runname, 0)

    for i in range(generations):
        print(f"\n ---- Run {runname}. Starting Gen {i}")

        if i > 0:
            # Main genetic algorithm code
            parents = select_best_score(population, all_q[i - 1], num_parents, reverse=True)
            offspring = crossover(parents, gen_size, elite, parents_perc)
            population = mutate(offspring)
            if elite:
                population = parents + population

            # Checkpoint
            if i % checkpoint == 0:
                visualize_graph_data(parents[:10], runname, i)
                plot_q(best_q, average_q, parents_q, runname)

        population_q = evaluate_q(population, normalize=False)
        record_q(population_q, all_q, best_q, average_q, parents_q, int(gen_size * parents_perc))
        print("Best Q: ", best_q[i])

    # Final operations
    plot_q(best_q, average_q, parents_q, runname)
    visualize_graph_data(parents[:10], runname, i)
    return best_q[i]


def compute_Qrand():
    gen_size = 1000
    population = [build_network() for i in range(gen_size)]
    population_q = evaluate_q(population, normalize=False)
    average_q = sum(population_q) / len(population_q)
    return average_q


if __name__ == "__main__":
    # QRand
    average_qrand = 0
    for i in range(100):
        Qrand = compute_Qrand()
        print(f"Simulation: {i} | Qrand: {Qrand}")
        average_qrand += Qrand
    average_qrand /= 100
    print("Final Qrand: ", average_qrand)

    # Qmax
    average_qmax = 0
    for i in range(100):
        Qmax = compute_Qmax(i)
        print(f"Simulation: {i} | Qmax: {Qmax}")
        average_qmax += Qmax
        # print("Final Qrand AGAIN IN CASE YOU  MISSED IT: ", average_qrand)
    average_qmax /= 100
    print("Final Qmax: ", average_qmax)






# Final Qrand:  0.010356569999999987
# Qmax: 0.82 (across ~4 simulations)



