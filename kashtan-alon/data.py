import csv
import time

import os
from os import makedirs
from os.path import join
from matplotlib import pyplot as plt
import subprocess

from visualize_nets import write_graphviz, plot_graphviz


def setup_savedir(runname):
    os.makedirs(join("network_graphs", runname).replace("\\", "/"), exist_ok=True)
    makedirs("loss_curves", exist_ok=True)


def record_loss(population_loss, all_losses, best_losses, average_losses):
    all_losses.append(population_loss)
    average_loss = round(sum(population_loss) / len(population_loss), 3)
    best_loss = round(min(population_loss), 3)
    best_losses.append(best_loss)
    average_losses.append(average_loss)


def save_networks(best_scores, average_scores, total_scores):
    with open('loss.csv', 'a') as fd:
        writer = csv.writer(fd)
        writer.writerow("new gen")
        writer.writerow(best_scores)
        writer.writerow(average_scores)
        writer.writerow(total_scores)


def plot_results(best_scores, average_scores, runname):
    fig = plt.figure(figsize=(21,7))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    # ax3 = fig.add_subplot(1, 3, 3)
    ax1.plot(best_scores, label='best loss')
    ax2.plot(average_scores, label='average loss')
    # ax3.plot(total_scores, label='total loss')
    ax1.set_xlabel('Generation (n)')
    ax1.set_ylabel('Loss')
    ax1.set_title('Best Loss Each Generation')
    ax1.legend()
    ax2.set_xlabel('Generation (n)')
    ax2.set_ylabel('Loss')
    ax2.set_title('Average Loss Each Generation')
    ax2.legend()
    # ax3.set_xlabel('Generation (n)')
    # ax3.set_ylabel('Loss')
    # ax3.set_title('Total Loss Each Generation')
    # ax3.legend()

    # plt.show()
    file_path = join('loss_curves', f'loss_{runname}.png').replace("\\", "/")
    plt.savefig(file_path)


def visualize_networks(population, runname, gen):
    num_to_plot = 0
    file_path = ""
    for i in range(len(population)):
        file_path = join('network_graphs', runname, f'graphviz_gen{gen}_model{i}.txt').replace("\\", "/")
        write_graphviz(population[i], file_path)
        plot_graphviz(file_path)


def visualize_solo_network(network, name=None):
    if name: name = name
    else: name = time.time()
    file_path = join('solo_networks', f"network_{name}.txt").replace("\\", "/")
    write_graphviz(network, file_path)
    plot_graphviz(file_path)












