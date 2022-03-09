"""
Various functions for visualizing different aspects of the experiement.
E.g. plotting the loss curves and graphing the evolved networks.
"""

import csv
import time

from os import makedirs
from os.path import join

import matplotlib
from matplotlib import pyplot as plt
import pickle
from tqdm import tqdm

from visualize_nets import write_graphviz, plot_graphviz, delete_graphviz_txts


def record_loss(population_loss, all_losses, best_losses, average_losses):
    all_losses.append(population_loss)
    average_loss = round(sum(population_loss) / len(population_loss), 3)
    best_loss = round(min(population_loss), 3)
    best_losses.append(best_loss)
    average_losses.append(average_loss)


def record_q(population_q, all_q, best_q, average_q):
    all_q.append(population_q)
    best_q.append(round(max(population_q), 3))
    average_q.append(round(sum(population_q) / len(population_q), 3))


def save_loss_to_csv(best_scores, average_scores, total_scores):
    with open('loss.csv', 'a') as fd:
        writer = csv.writer(fd)
        writer.writerow("new gen")
        writer.writerow(best_scores)
        writer.writerow(average_scores)
        writer.writerow(total_scores)


def plot_loss(best_scores, average_scores, runname):
    matplotlib.use("Agg")
    fig = plt.figure(figsize=(24,8))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.plot(best_scores, label='best loss')
    ax2.plot(average_scores, label='average loss')
    ax1.set_xlabel('Generation (n)')
    ax1.set_ylabel('Loss')
    ax1.set_title('Best Loss Each Generation')
    ax1.legend()
    ax2.set_xlabel('Generation (n)')
    ax2.set_ylabel('Loss')
    ax2.set_title('Average Loss Each Generation')
    ax2.legend()

    # plt.show()
    file_path = join('loss_curves', f'loss_{runname}').replace("\\", "/")
    plt.savefig(file_path+".png")
    with open(file_path+".pickle", 'wb') as fid:
        pickle.dump(fig, fid)

    fig.clear()
    plt.close(fig)


def plot_q(best_scores, average_scores, runname):
    matplotlib.use("Agg")
    fig = plt.figure(figsize=(24,8))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.plot(best_scores, label='best Q')
    ax2.plot(average_scores, label='average Q')
    ax1.set_xlabel('Generation (n)')
    ax1.set_ylabel('Q')
    ax1.set_title('Best Q Each Generation')
    ax1.legend()
    ax2.set_xlabel('Generation (n)')
    ax2.set_ylabel('Q')
    ax2.set_title('Average Q Each Generation')
    ax2.legend()

    file_path = join('Q_curves', f'Q_{runname}').replace("\\", "/")
    plt.savefig(file_path+".png")
    with open(file_path+".pickle", 'wb') as fid:
        pickle.dump(fig, fid)

    fig.clear()
    plt.close(fig)


def visualize_networks(population, runname, gen):
    file_path = ""
    for i in tqdm(range(len(population)), desc="Saving plots of networks"):
        makedirs(join('graphviz_plots', runname, f"gen_{gen}").replace("\\", "/"), exist_ok=True)
        folder_path = join('graphviz_plots', runname, f"gen_{gen}").replace("\\", "/")
        file_path = join(folder_path, f'graphviz_model{i}.txt').replace("\\", "/")
        write_graphviz(population[i], file_path)
        plot_graphviz(file_path)
        delete_graphviz_txts(folder_path)


def visualize_solo_network(network, name=None):
    if name: name = name
    else: name = time.time()
    makedirs('iso_networks', exist_ok=True)
    file_path = join('iso_networks', f"{name}.txt").replace("\\", "/")
    write_graphviz(network, file_path)
    plot_graphviz(file_path)












