"""
Measures broadness
"""

from main import build_network, generate_samples, mutate, load_weights, evaluate_population, count_connections
import pandas as pd


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
        print(f"\n Simulation {s}")
        population = [load_weights(initial_gen_path)[0]]
        # population = [build_network() for i in range(gen_size)]

        for i in range(mutation_steps):
            print(f"\n ---- Run {runname}. Mutation Step {i}")
            if goal_is_and: print(f"Goal is L AND R")
            else: print("Goal is L OR R")

            loss = evaluate_population(population, samples, goal_is_and, loss="loss", activation="tanh")[0]
            losses_in_simulation.append(loss)
            print("Loss: ", loss)

            population = mutate(population)
            print("Connection: ", count_connections(population[0]))

        # all_losses.append(losses_in_simulation)

    sim_num = [j+1 for j in range(simulations) for i in range(mutation_steps)]
    steps = list(range(1,mutation_steps+1)) * simulations
    delta_l = [y_i-x_i for y_i,x_i in zip(losses_in_simulation[1:], losses_in_simulation)]
    delta_l.insert(0,0)
    ave_delta_per_step = [round(i/(n+1),3) for i,n in zip(losses_in_simulation, list(range(int(len(losses_in_simulation) / simulations))) * simulations )]
                         # [i / (n + 1) for i, n in zip(l, list(range(int(len(l) / 2))) * 2)]

    dict = {"Simulation Num": sim_num, "Steps": steps, "Loss": losses_in_simulation, "Delta Loss this step": delta_l, "Average Delta Loss per n Steps": ave_delta_per_step}
    df = pd.DataFrame(dict)
    df.to_csv(f"broadness/{runname}.csv")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()







