import numpy as np
import random
from generate_labeled_data import generate_samples, save_samples, load_samples, filter_samples
from neural_network import evaluate_population


luc_samples = generate_samples(256)
filtered_samples = filter_samples(luc_samples, [3])

theta1 = np.array([ [1, 0], [1, 0], [1, 0], [1, 0],
                   [0, 1],[0, 1],[0, 1],[0, 1]] )
theta2 = np.array([ [1],
                    [1]] )
thrsh1 = np.ones((2, 1))
thrsh2 = np.ones((1, 1))

thetas = [theta1, theta2]
thresholds = [thrsh1, thrsh2]

network = {"thetas": thetas, "thresholds": thresholds, "loss": 0}

loss = evaluate_population([network], filtered_samples, False)
print("loss ", loss)
# thr1 = np.array([0,1,0,2,0,1]).reshape(6,1)
# thr2 = np.array([2,2,0,2]).reshape(4,1)
# thr3 = np.array([1,1]).reshape(2,1)
# thr4 = np.array([1]).reshape(1,1)
#
# theta1 = np.array([[-1,-1,-1,0,0,0],[-1,-1,-1,0,0,0],[0,0,1,0,0,0],[1,-1,0,0,0,0],
#                    [0,0,0,0,1,-1],[0,0,0,1,0,0],[0,0,0,1,-1,-1],[0,0,0,1,1,1]])
# theta2 = np.array([[2,0,0,0],[-1,-2,0,0],[0,-1,0,0],
#                    [0,0,-1,1],[0,0,-2,1],[0,0,0,-1]])
# theta3 = np.array([[-1,0],[2,0],
#                    [0,2],[0,1]])
# theta4 = np.array([[1],[1]])
#
# z1 = np.dot(x.transpose(), theta1)
# z2 = np.dot(z1, theta2)
# z3 = np.dot(z2, theta3)
# z4 = np.dot(z3, theta4)
#
# print(z1, z2, z3, z4)
# y = 2


# thr1 = np.random.randint(-4,3, (8,1))
# thr2 = np.random.randint(-4,3, (4,1))
# thr3 = np.random.randint(-4,3, (2,1))
# thr4 = np.random.randint(-2,1, (1,1))

# theta1 = np.random.choice([-1,1], (8,8))
# theta2 = np.random.choice([-1,1], (8,4))
# theta3 = np.random.choice([-1,1], (4,2))
# theta4 = np.random.choice([-1,1], (2,1))

# temp0 = np.array([[1,2,3,4,5,6,7,8]])
# temp1 = np.zeros((7,8))
#
# theta1 = np.concatenate((temp0, temp1))
# theta2 = np.ones((8,4))
# theta3 = np.array([[1,0],[1,0],[-1,-1],[1,1]])
# theta4 = np.array([[1],[-3]])
#
# ts = [theta1, theta2, theta3, theta4]


def apply_neuron_constraints(thetas):
    for theta in thetas:
        theta = theta.transpose()
        for node_num in range(len(theta)):
            total = sum(abs(theta[node_num]))
            while total > 3:
                choice = random.randint(0,len(theta.transpose())-1)
                theta[node_num][choice] = 0
                total = sum(abs(theta[node_num]))


# apply_neuron_constraints(ts)

# z1 = np.dot(x.transpose(), theta1)
# z2 = np.dot(z1, theta2)
# z3 = np.dot(z2, theta3)
# z4 = np.dot(z3, theta4)
# print(z1, z2, z3, z4)

# z = np.dot(x2.transpose(), ts[0])
# for theta in ts[1:]:
#     z = np.dot(z, theta)
#     print(z)

y = 2
