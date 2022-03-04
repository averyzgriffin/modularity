"""
Functions for saving various aspects of the experiement.
E.g. saving weights, plots, graphs
"""

import csv
import time

import json
from os import makedirs
from os.path import join
from matplotlib import pyplot as plt
import numpy as np
import pickle
from tqdm import tqdm

from visualize_nets import write_graphviz, plot_graphviz


# def setup_savedir(runname):
#     makedirs(join("network_graphs", runname).replace("\\", "/"), exist_ok=True)
#     makedirs("loss_curves", exist_ok=True)
#     makedirs(f"saved_weights/{runname}", exist_ok=True)

def default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError('Not serializable')


def save_weights(population, runname, gen):
    makedirs(f"saved_weights/{runname}/gen_{gen}", exist_ok=True)
    for i in range(len(population)):
        w_file = open(f"saved_weights/{runname}/gen_{gen}/network_{i}.json", "w")
        json.dump(population[i], w_file, default=default)
        w_file.close()


def save_q(best_q, runname):
    makedirs(f"saved_qvalues", exist_ok=True)
    w_file = open(f"saved_qvalues/{runname}.txt", "w")
    w_file.write(str(best_q))
    w_file.close()
