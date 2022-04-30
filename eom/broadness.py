"""
Measures broadness
"""
import copy
from main import build_network, generate_samples, mutate, load_weights, evaluate_population
from perfect_modularity import build_perfect_network
import pandas as pd


def get_average(i, n):
    try:
        return round(i / (n), 3)
    except ZeroDivisionError:
        return 0

def main():
    runname = "mvg_broadness_005"
    initial_gen_path = "saved_weights/mvg_gen1000_005/gen_5000"
    goal_is_and = False
    mutation_steps = 100
    simulations = 100

    all_losses = []
    samples = generate_samples()
    losses_in_simulation = []

    for s in range(simulations):
        # population = [load_weights(initial_gen_path)[0]]
        population = [build_perfect_network()]

        for i in range(mutation_steps):
            print(f"\n ---- Run {runname}. Simulation # {s}. Mutation Step {i}")
            if goal_is_and: print(f"Goal is L AND R")
            else: print("Goal is L OR R")

            loss = evaluate_population(population, samples, goal_is_and, loss="loss", activation="tanh")[0]
            losses_in_simulation.append(loss)
            print("Loss: ", loss)

            population = mutate(population, old_way=True)

        # all_losses.append(losses_in_simulation)

    sim_num = [j+1 for j in range(simulations) for i in range(mutation_steps)]
    steps = list(range(mutation_steps)) * simulations
    delta_l = [y_i-x_i for y_i,x_i in zip(losses_in_simulation[1:], losses_in_simulation)]
    delta_l.insert(0,0)
    ave_delta_per_step = [get_average(i,n) for i,n in zip([x - losses_in_simulation[0] for x in losses_in_simulation], list(range(int(len(losses_in_simulation) / simulations))) * simulations )]

    # ne = [x for y in (ave_delta_per_step[i:i + mutation_steps] + [0] * (i < len(ave_delta_per_step) - mutation_steps-1) for i in range(0, len(ave_delta_per_step), mutation_steps)) for x in y]
    # [i / (n + 1) for i, n in zip(l, list(range(int(len(l) / 2))) * 2)]

    dict = {"Simulation Num": sim_num, "Steps": steps, "Loss": losses_in_simulation, "Delta Loss this step": delta_l, "Average Delta Loss per n Steps": ave_delta_per_step}
    df = pd.DataFrame(dict)
    df.to_csv(f"broadness/{runname}.csv")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()







