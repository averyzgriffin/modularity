import numpy as np
from generate_labeled_data import generate_samples
from neural_network import evaluate_population
from data_viz import record_loss, visualize_solo_network
import json


def default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError('Not serializable')


theta1 = np.array([[-1,1,0,0,0,0,0,0],
                   [-1,0,1,0,0,-1,0,0],
                   [0,0,2,1,0,0,0,0],
                   [0,0,0,2,1,0,0,0],
                   [-1,-1,0,0,-1,0,-1,-2],
                   [0,0,0,0,0,0,0,0],
                   [0,0,0,0,1,0,0,0],
                   [0,0,0,0,0,-2,1,-1]])
# theta1 = np.array([[2,1,0,0,0,0,0,0],
#                    [-1,0,1,-2,0,0,0,0],
#                    [0,1,0,0,1,0,0,0],
#                    [0,-1,-1,0,0,0,-1,-1],
#                    [0,0,0,1,1,0,0,0],
#                    [0,0,0,0,1,2,0,0],
#                    [0,0,0,0,0,1,0,-1],
#                    [0,0,0,0,0,0,1,-1]])
theta2 = np.array([[1,0,0,0],
                   [1,0,0,0],
                   [-1,-1,1,0],
                   [0,0,0,0],
                   [0,0,0,0],
                   [0,2,-1,0],
                   [0,0,1,-2],
                   [0,0,0,-1]])
# theta2 = np.array([[0,0,0,0],
#                    [-1,1,0,0],
#                    [-2,-1,0,0],
#                    [0,0,2,0],
#                    [0,0,0,0],
#                    [0,1,-1,-1],
#                    [0,0,0,1],
#                    [0,0,0,1]])
theta3 = np.array([[1,1],
                   [-1,0],
                   [0,0],
                   [0,-2]])
# theta3 = np.array([[-2,0],
#                    [0,0],
#                    [0,-2],
#                    [1,1]])
theta4 = np.array([[-2],
                   [-1]])
# theta4 = np.array([[1],
#                    [-2]])
thrsh1 = np.array([[-2],[0],[2],[1],[2],[1],[3],[2]])
thrsh2 = np.array([[0],[-3],[-1],[1]])
thrsh3 = np.array([[-2],[3]])
thrsh4 = np.array([[-1]])

thetas = [theta1, theta2, theta3, theta4]
thresholds = [thrsh1, thrsh2, thrsh3, thrsh4]

network = {"thetas": thetas, "thresholds": thresholds, "loss": 0}
population = [network]

samples = generate_samples(100)
for i in range(len(samples)):
    w_file = open(f"samples/sample_{i}.json", "w")
    json.dump(samples[i], w_file, default=default)
    w_file.close()

loaded_samples = []
for i in range(len(samples)):
    w_file = open(f"samples/sample_{i}.json", "r")
    sample = json.load(w_file)
    loaded_samples.append(sample)
    w_file.close()

# visualize_solo_network(population[0], "reconstructed")

for i in range(100):
    samples = generate_samples(100)
    loss = evaluate_population(population, samples, 1, False)
    print("Loss: ", loss)

check = 4