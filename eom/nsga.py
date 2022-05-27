from collections import defaultdict
import numpy as np
import random

population_size = 10
min_value = -10
max_value = 10
num_parents = max(int(population_size*.2), 2)
r = .5


# def main():
#     population = initialize_pop(population_size)
#
#     for i in range(100):
#         losses = compute_performance_loss(population)
#         connection_costs = compute_connection_count_costs(population)  # eventually, compute_connection_distance_costs
#         parents = stochastic_dominant_selection(population, r, num_parents)  # use crowding_distance() to break ties
#         offspring = crossover(parents, N)
        # offspring = mutate(offspring)
        # population = population + offspring
        # population = nondomimant_sorting(population)  # use crowding_distance() to sort any overflowing fronts


def main2():
    population = initialize_pop(population_size)
    for i in range(100):
        pop_objs = [tuple([obj_1(i), obj_2(i)]) for i in population]
        distances = CrowdingDist(pop_objs)

        parents = stochastic_dominant_selection(population, pop_objs, r, num_parents, distances)
        offspring = crossover_and_mutate(parents, population_size)
        offspring_objs = [tuple([obj_1(i), obj_2(i)]) for i in offspring]
        offspring_distances = CrowdingDist(offspring_objs)

        merged_pop = population + offspring
        merged_objs = pop_objs + offspring_objs
        merged_distances = distances + offspring_distances

        fronts = sortNondominated(merged_objs)
        population = selectPopulationFromFronts(fronts, merged_pop, merged_distances)

        min_loss = min(merged_objs[0])
        print(i)
        print(population)
        print(pop_objs)


def initialize_pop(pop_size):
    x = random.sample(range(min_value, max_value), pop_size)
    # y = [{"id": str(i), "genome": j, "obj1": None, "obj2": None} for i,j in zip(range(len(x)), x)]
    return x


def stochastic_dominant_selection(population, pop_objs, r, num_parents, distances):
    parents_idxs = []
    while len(parents_idxs) < num_parents:
        ps = random.sample(range(len(pop_objs)), 2)
        parent1 = pop_objs[ps[0]]
        parent2 = pop_objs[ps[1]]

        stochastic = np.random.uniform(0, 1) <= r
        if not stochastic:
            if outperforms(parent1[0], parent2[0]):
                parents_idxs.append(ps[0])
            elif outperforms(parent2[0], parent1[0]):
                parents_idxs.append(ps[1])
            else:
                parents_idxs.append(less_crowded(distances, ps[0], ps[1]))
        elif stochastic:
            if dominates(parent1, parent2, [-1, -1]):
                parents_idxs.append(ps[0])
            elif dominates(parent2, parent1, [-1, -1]):
                parents_idxs.append(ps[1])
            else:
                parents_idxs.append(less_crowded(distances, ps[0], ps[1]))
                # if distances[ps[0]] > distances[ps[1]]:
                #     parents_idxs.append(ps[0])
                # elif distances[ps[0]] < distances[ps[1]]:
                #     parents_idxs.append(ps[1])
                # else:
                #     parents_idxs.append(current_parents[random.randint(0, 1)][0])

    return [population[i] for i in parents_idxs]


def less_crowded(distances, a, b):
    if distances[a] > distances[b]:
        return a
    elif distances[a] < distances[b]:
        return b
    else:
        return random.choice([a, b])


def outperforms(a, b, sign=-1):
    if a * sign > b * sign:
        return True
    else:
        return False


def dominates(objs_orgA, objs_orgB, sign=[1, 1]):
    """ Returns True iff organism A dominates organism B along all objectives"""
    indicator = False
    for a, b, sign in zip(objs_orgA, objs_orgB, sign):
        if a * sign > b * sign:
            indicator = True
        # if a is dominated by b in one of the objectives, then return False
        elif a * sign < b * sign:
            return False
    return indicator


def obj_1(x):
    y = -x ** 3
    return y


def obj_2(x):
    y = -(x-2)**2
    return y


def CrowdingDist(fitness=None):
    """
    :param fitness: A list of fitness values
    :return: A list of crowding distances of chrmosomes

    The crowding-distance computation requires sorting the population according to each objective function value
    in ascending order of magnitude. Thereafter, for each objective function, the boundary solutions (solutions with smallest and largest function values)
    are assigned an infinite distance value. All other intermediate solutions are assigned a distance value equal to
    the absolute normalized difference in the function values of two adjacent solutions.
    """

    # initialize list: [0.0, 0.0, 0.0, ...]
    distances = [0.0] * len(fitness)
    crowd = [(f_value, i) for i, f_value in enumerate(fitness)]  # create keys for fitness values

    n_obj = len(fitness[0])

    for i in range(n_obj):  # calculate for each objective
        crowd.sort(key=lambda element: element[0][i])
        # After sorting,  boundary solutions are assigned Inf
        # crowd: [([obj_1, obj_2, ...], i_0), ([obj_1, obj_2, ...], i_1), ...]
        distances[crowd[0][1]] = float("Inf")
        distances[crowd[-1][1]] = float("inf")
        if crowd[-1][0][i] == crowd[0][0][i]:  # If objective values are same, skip this loop
            continue
        # normalization (max - min) as Denominator
        norm = float(crowd[-1][0][i] - crowd[0][0][i])
        # crowd: [([obj_1, obj_2, ...], i_0), ([obj_1, obj_2, ...], i_1), ...]
        # calculate each individual's Crowding Distance of i th objective
        # technique: shift the list and zip
        for prev, cur, next in zip(crowd[:-2], crowd[1:-1], crowd[2:]):
            distances[cur[1]] += (next[0][i] - prev[0][i]) / norm  # sum up the distance of ith individual along each of the objectives

    return distances


def sortNondominated(individuals, k=None, first_front_only=False):
    """Sort the first *k* *individuals* into different nondomination levels
        using the "Fast Nondominated Sorting Approach" proposed by Deb et al.,
        see [Deb2002]_. This algorithm has a time complexity of :math:`O(MN^2)`,
        where :math:`M` is the number of objectives and :math:`N` the number of
        individuals.
        :param individuals: A list of individuals to select from.
        :param k: The number of individuals to select.
        :param first_front_only: If :obj:`True` sort only the first front and
                                    exit.
        :param sign: indicate the objectives are maximized or minimized
        :returns: A list of Pareto fronts (lists), the first list includes
                    nondominated individuals.
        .. [Deb2002] Deb, Pratab, Agarwal, and Meyarivan, "A fast elitist
            non-dominated sorting genetic algorithm for multi-objective
            optimization: NSGA-II", 2002.
    """
    if k is None:
        k = len(individuals)

    map_fit_ind = defaultdict(list)
    for i, f_value in enumerate(individuals):  # fitness = [(1, 2), (2, 2), (3, 1), (1, 4), (1, 1)...]
        map_fit_ind[f_value].append(i)
    fits = list(map_fit_ind.keys())  # fitness values

    current_front = []
    next_front = []
    dominating_fits = defaultdict(int)  # n (The number of people dominate you)
    dominated_fits = defaultdict(list)  # Sp (The people you dominate)

    # Rank first Pareto front
    # *fits* is a iterable list of chromosomes. Each has multiple objectives.
    for i, fit_i in enumerate(fits):
        for fit_j in fits[i + 1:]:
            # Eventhougn equals or empty list, n & Sp won't be affected
            if dominates(fit_i, fit_j):
                dominating_fits[fit_j] += 1
                dominated_fits[fit_i].append(fit_j)
            elif dominates(fit_j, fit_i):
                dominating_fits[fit_i] += 1
                dominated_fits[fit_j].append(fit_i)
        if dominating_fits[fit_i] == 0:
            current_front.append(fit_i)

    fronts = [[]]  # The first front
    for fit in current_front:
        fronts[-1].extend(map_fit_ind[fit])
    pareto_sorted = len(fronts[-1])

    # Rank the next front until all individuals are sorted or
    # the given number of individual are sorted.
    # If Sn=0 then the set of objectives belongs to the next front
    if not first_front_only:  # first front only
        N = min(len(individuals), k)
        while pareto_sorted < N:
            fronts.append([])
            for fit_p in current_front:
                # Iterate Sn in current fronts
                for fit_d in dominated_fits[fit_p]:
                    dominating_fits[fit_d] -= 1  # Next front -> Sn - 1
                    if dominating_fits[fit_d] == 0:  # Sn=0 -> next front
                        next_front.append(fit_d)
                         # Count and append chromosomes with same objectives
                        pareto_sorted += len(map_fit_ind[fit_d])
                        fronts[-1].extend(map_fit_ind[fit_d])
            current_front = next_front
            next_front = []

    return fronts


def selectPopulationFromFronts(fronts, population, distances):
    idxs = []
    for front in fronts:
        if len(idxs) + len(front) <= population_size:
           idxs.extend(front)
        else:
            d = population_size - len(idxs)
            temp = []
            for id in front:
                temp.append((distances[id], id))
            temp.sort(reverse=True)
            front = [thing[1] for thing in temp]
            idxs.extend(front[:d])
        if len(idxs) == population_size:
            break

    return [population[i] for i in idxs]


def crossover_and_mutate(parents, population_size):
    offspring = []
    while len(offspring) < population_size:
        ps = random.sample(range(len(parents)), 2)
        p1 = parents[ps[0]]
        p2 = parents[ps[1]]

        r=random.random()
        if r>0.5:
            offspring.append(int(mutate((p1+p2)/2)))
        else:
            offspring.append(int(mutate((p1-p2)/2)))
    return offspring


def mutate(solution):
    mutation_prob = random.random()
    if mutation_prob < .1:
        solution = random.randint(-10,10)
    return solution


main2()















