import numpy as np
from generate_labeled_data import generate_samples, load_samples
from neural_network import evaluate_population
from data_viz import record_loss, visualize_solo_network
import json


def default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError('Not serializable')


# theta1 = np.array([[-1,-1,1,0,0,0,0,0],
#                    [-1,1,1,0,1,0,0,0],
#                    [1,1,0,2,0,1,0,0],
#                    [0,0,0,1,1,0,-1,0],
#                    [0,0,0,0,0,0,0,-1],
#                    [0,0,0,0,0,0,0,0],
#                    [0,0,0,0,-1,0,0,0],
#                    [0,0,-1,0,0,1,2,2]])
# # theta1 = np.array([[2,1,0,0,0,0,0,0],
# #                    [-1,0,1,-2,0,0,0,0],
# #                    [0,1,0,0,1,0,0,0],
# #                    [0,-1,-1,0,0,0,-1,-1],
# #                    [0,0,0,1,1,0,0,0],
# #                    [0,0,0,0,1,2,0,0],
# #                    [0,0,0,0,0,1,0,-1],
# #                    [0,0,0,0,0,0,1,-1]])
# theta2 = np.array([[1,-2,0,0],
#                    [0,0,0,0],
#                    [-1,0,-2,0],
#                    [0,0,0,-2],
#                    [0,0,0,0],
#                    [-1,0,0,0],
#                    [0,0,0,0],
#                    [0,-1,1,1]])
# # theta2 = np.array([[0,0,0,0],
# #                    [-1,1,0,0],
# #                    [-2,-1,0,0],
# #                    [0,0,2,0],
# #                    [0,0,0,0],
# #                    [0,1,-1,-1],
# #                    [0,0,0,1],
# #                    [0,0,0,1]])
# theta3 = np.array([[2,0],
#                    [0,1],
#                    [0,0],
#                    [1,-1]])
# # theta3 = np.array([[-2,0],
# #                    [0,0],
# #                    [0,-2],
# #                    [1,1]])
# theta4 = np.array([[-1],
#                    [1]])
# # theta4 = np.array([[1],
# #                    [-2]])
# thrsh1 = np.array([[0],[-4],[-1],[0],[-4],[0],[-2],[3]])
# thrsh2 = np.array([[-2],[-1],[1],[-3]])
# thrsh3 = np.array([[1],[-4]])
# thrsh4 = np.array([[0]])
#
# thetas = [theta1, theta2, theta3, theta4]
# thresholds = [thrsh1, thrsh2, thrsh3, thrsh4]
#
# network = {"thetas": thetas, "thresholds": thresholds, "loss": 0}

# samples = generate_samples(100)
# for i in range(len(samples)):
#     w_file = open(f"samples/sample_{i}.json", "w")
#     json.dump(samples[i], w_file, default=default)
#     w_file.close()

# loaded_samples = []
# for i in range(len(samples)):
#     w_file = open(f"samples/sample_{i}.json", "r")
#     sample = json.load(w_file)
#     loaded_samples.append(sample)
#     w_file.close()

def recurse(item, key):
    if isinstance(item[key], list):
        item = np.array(item[key])
        for subitem in item:
            recurse(subitem)


def make_array(network):
    keys = ["thetas", "thresholds"]
    for key in keys:
        recurse(network, key)
    return network


def the_hard_way(network):
    keys = ["thetas", "thresholds"]
    for key in keys:
        for i in range(len(network[key])):
            for j in range(len(network[key][i])):
                if isinstance(network[key][i][j], list):
                    network[key][i][j] = np.array(network[key][i][j])
            if isinstance(network[key][i], list):
                network[key][i] = np.array(network[key][i])

population = []
for i in range(100):
    w_file = open(f"saved_weights/network_{i}.json", "r")
    network = json.load(w_file)
    the_hard_way(network)
    population.append(network)
    w_file.close()


samples = load_samples(100, "samples")
# for n in range(len(population)):
#     visualize_solo_network(population[n], f"reconstructed_{n}")
loss = evaluate_population(population, samples, 1, False)
print("Loss: ", loss)

check = 4