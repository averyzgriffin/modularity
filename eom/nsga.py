import numpy as np
import random

N = population_size = 25
N = offspring_size
r = stochastic_prob
num_parents = population_size*.4


def main():

    population = initialize_pop()

    for i in range(100):
        parents = stochastic_dominant_selection(population, r, num_parents)  # use crowding_distance() to break ties
        offspring = crossover(parents, N)
        offspring = mutate(offspring)
        population = population + offspring
        new_population = nondomimant_sorting(population)
            # use crowding_distance() to sort any overflowing fronts


def initialize_pop():
    return random.sample(range(-100, 100), 25)


def obj_1(x):
    y = -x ** 3
    return y


def obj_2(x):
    y = -(x-2)**2
    return y


def stochastic_dominant_selection(population, r, num_parents):
    parents = []
    while len(parents) < num_parents:
        ps = random.sample(range(len(population)), 2)
        parent1 = population[ps[0]]
        parent2 = population[ps[1]]
        # TODO Figure out how to structure individuals
        if (np.random.uniform(0,1) > r and outperforms(parent1["loss"], parent2["loss"])) or dominates(parent1["objs"], parent2["objs"], [-1, 1]):
            parents.append(parent1)
        elif ( np.random.uniform(0,1) > r and outperforms(parent2["loss"], parent1["loss"]) ) or dominates(parent2["objs"], parent1["objs"], [-1, 1]):
            parents.append(parent2)

        stochastic = np.random.uniform(0, 1) <= r

        if not stochastic:
            if outperforms(parent1["loss"], parent2["loss"]):
                parents.append(parent1)
            elif outperforms(parent2["loss"], parent1["loss"]):
                parents.append(parent2)
            else:
                parents.append([parent1, parent2][less_crowded(parent1, parent2)])
        elif stochastic:
            if dominates(parent1["objs"], parent2["objs"], [-1, 1]):
                parents.append(parent1)
            elif dominates(parent2["objs"], parent1["objs"], [-1, 1]):
                parents.append(parent2)
            else:
                parents.append([parent1, parent2][less_crowded(parent1, parent2)])

    return parents


def less_crowded(a, b):
    # TODO Write this function
    return 1


def outperforms(a, b, sign=-1):
    if a * sign > b * sign:
        return True
    else:
        return False

def dominates(objs_orgA, objs_orgB, sign=[1, 1]):
    indicator = False
    for a, b, sign in zip(obj1, obj2, sign):
        if a * sign > b * sign:
            indicator = True
        # if one of the objectives is dominated, then return False
        elif a * sign < b * sign:
            return False
    return indicator































