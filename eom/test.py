import numpy as np
import random
from timeit import default_timer as timer


def A():
    population = initialize_popA(100)
    for i in range(100):
        orgs = [tuple([obj_1(i), obj_2(i)]) for i in population]

def B():
    population = initialize_popB(100)
    for i in range(100):
        orgs = []
        for ind in population:
            ind["obj1"] = obj_1(ind["genome"])
            ind["obj2"] = obj_2(ind["genome"])
            orgs.append(tuple([ind["obj1"], ind["obj2"]]))

def initialize_popA(pop_size):
    x = random.sample(range(-100, 100), pop_size)
    return x

def initialize_popB(pop_size):
    x = random.sample(range(-100, 100), pop_size)
    y = [{"id": str(i), "genome": j, "obj1": None, "obj2": None} for i,j in zip(range(len(x)), x)]
    return y

def obj_1(x):
    y = -x ** 3
    return y

def obj_2(x):
    y = -(x-2)**2
    return y
# def timing():
#     start = timer()
#     A()
#     end = timer()
#     print("A", end - start)
#
#     start = timer()
#     B()
#     end = timer()
#     print("B", end - start)


if __name__ == "__main__":
    import timeit
    print(timeit.timeit("B()", setup="from __main__ import B", number=1000))
