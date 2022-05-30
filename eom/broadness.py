"""
Measures broadness
"""
from csv import writer
import copy
import warnings
from os import path

from main import build_network, generate_samples, mutate, load_weights, evaluate_population, makedirs
# from perfect_modularity import build_perfect_network
import numpy as np
import pandas as pd
from tqdm import tqdm


def get_average(i, n):
    try:
        return round(i / (n), 3)
    except ZeroDivisionError:
        return 0


def start():
    supertrial = "051122_SuperEvolve_MVG"
    trial_type = "mvg"
    goal_is_and = False
    version = "best"
    print("\nMeasuring broadenss for: ", supertrial)

    makedirs(f"measuring_broadness/050122_Supertrial/{supertrial}", exist_ok=True)

    path = f"csvs/{supertrial}/{supertrial}_combined.csv"
    df = pd.read_csv(path)
    for idx, row in df.iterrows():
        if idx == 11:
            trial_num = int(row["Trial #"])
            gen = 573
            starting_loss = row["Best Loss"]
            population = load_weights(f"saved_weights/{supertrial}/trial_{str(trial_num).zfill(3)}/gen_{gen}")
            samples = generate_samples()
            pool = []

            if trial_type == "mvg":
                goal_is_and = get_goal(gen)

            for i in range(len(population)):
                print("Searching through population for best net", i)
                network = population[i]
                loss = evaluate_population([network], samples, goal_is_and, "loss", activation="tanh")
                if loss[0] == starting_loss:
                    pool.append(network)
                    break

            if len(pool) == 1:
                main(pool, goal_is_and, trial_type, starting_loss, gen, trial_num, supertrial, version)
            else: warnings.warn("Pool did not contain 1 element")


def get_goal(gen):
    i = 0
    while gen > 20:
        gen -= 20
        i += 1
    if i % 2 == 0:
        return True
    return False


def main(population, goal_is_and, trial_type, starting_loss, gen_num, trial_num, supertrial, version):
    mutation_steps = 4
    simulations = 1000
    goal = "OR"
    if goal_is_and:
        goal = "AND"

    print("\nTrial # ", trial_num)
    print("Starting Loss: ", starting_loss)
    print("Goal is ", goal)

    samples = generate_samples()
    losses_in_simulation = []

    # network = build_perfect_network()
    # network = load_weights(initial_gen_path)[0]

    for network in population:
        for s in tqdm(range(simulations)):
            population = [copy.deepcopy(network)]

            for i in range(mutation_steps):
                loss = evaluate_population(population, samples, goal_is_and, loss="loss", activation="tanh")[0]
                losses_in_simulation.append(loss)
                population = mutate(population, broadness=True)

        sim_num = [j+1 for j in range(simulations) for i in range(mutation_steps)]
        steps = list(range(mutation_steps)) * simulations
        delta_l = [y_i-x_i for y_i,x_i in zip(losses_in_simulation[1:], losses_in_simulation)]
        delta_l.insert(0,0)
        ave_delta_per_step = [get_average(i,n) for i,n in zip([x - losses_in_simulation[0] for x in losses_in_simulation], list(range(int(len(losses_in_simulation) / simulations))) * simulations )]

        xbar = []
        var =  []
        for i in range(1,mutation_steps):
            y = ave_delta_per_step[i::mutation_steps]
            xbar.append(np.mean(y))
            var.append(np.var(y))
        print("Xbar: ", xbar)
        print("var: ", var)

        dict = {"Simulation Num": sim_num, "Steps": steps, "Loss": losses_in_simulation, "Delta Loss this step": delta_l, "Average Delta Loss per n Steps": ave_delta_per_step}
        df = pd.DataFrame(dict)
        df.to_csv(f"measuring_broadness/050122_Supertrial/{supertrial}/trial_{str(trial_num).zfill(3)}_{version}gen.csv", index=False)

        output_contents = [trial_type, goal, trial_num, starting_loss, gen_num, xbar[0], xbar[1], xbar[2], var[0], var[1], var[2]]
        column_labels = ["Trial Type", "Goal", "Trial #", "Loss", "Gen #", "Mean-Average Loss 1 Step", "MAL 2 Steps",
                         "MAL 3 Steps", "Variance-Average Loss 1 Step", "VAL 2 Steps", "VAL 3 Steps"]
        csv_file = f"measuring_broadness/050122_Supertrial/master.csv"

        if path.isfile(csv_file): mode = "a"
        else: mode = "w"
        try:
            with open(csv_file, mode, newline='') as write_obj:
                csv_writer = writer(write_obj)
                if mode == "w":
                    csv_writer.writerow(column_labels)
                csv_writer.writerow(output_contents)
        except Exception as e:
            print(e)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    start()







