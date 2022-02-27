from autograd import numpy as np, grad
from scipy.optimize import minimize
from matplotlib import pyplot
import time

def nand(a, b):
    # Only meant to be used for inputs between 0 and 1
    return 1 - a*b*(2-a)*(2-b)

def eval_circuit(a_weights, b_weights, inputs):
    m = a_weights.shape[0]
    n = a_weights.shape[1] - m
    
    #print(a_weights.shape, b_weights.shape, inputs.shape)
    
    state = []
    for i in range(m):
        a = np.dot(a_weights[i, :n], inputs) + np.dot(a_weights[i, n:n+i], np.array(state))
        b = np.dot(b_weights[i, :n], inputs) + np.dot(b_weights[i, n:n+i], np.array(state))
        state.append(nand(a, b))
    return state[-1]
'''
def test_eval_circuit():
    # Fixed-point test on a line; should return logical 1
    weights = np.eye(10, 11)
    inputs = np.ones((1,5))
    return eval_circuit(weights, inputs)
'''
def target_fcn_1(x):
    return (x[0] ^ x[1]) & (x[2] ^ x[3])

def target_fcn_2(x):
    return (x[0] ^ x[1]) | (x[2] ^ x[3])

#def fit_switching(N=1000, n=20):
N=1000; n=20

def obj(flatpack):
    raw_weights = np.tril(np.reshape(flatpack, (2, n, n+6)), k=5)**2
    a_weights = (raw_weights[0].T/np.sum(raw_weights[0], 1)).T
    b_weights = (raw_weights[1].T/np.sum(raw_weights[1], 1)).T
    predictions = eval_circuit(a_weights, b_weights, full_data)
    return 0.5*np.sum((predictions - z_data)*(predictions - z_data))

x_data = (np.random.random((4, N)) < 0.5)
z_data = target_fcn_2(x_data)
full_data = np.array([x_line for x_line in x_data] + [np.ones(N), np.zeros(N)])
def check_ideal():
    weights = np.zeros((2, n, n+6))
    
    weights[0, 0, 0] = 1.0
    weights[1, 0, 1] = 1.0
    weights[0, 1, 0] = 1.0
    weights[1, 1, 6] = 1.0
    weights[0, 2, 1] = 1.0
    weights[1, 2, 6] = 1.0
    weights[0, 3, 7] = 1.0
    weights[1, 3, 8] = 1.0
    
    weights[0, 4, 2] = 1.0
    weights[1, 4, 3] = 1.0
    weights[0, 5, 2] = 1.0
    weights[1, 5, 10] = 1.0
    weights[0, 6, 3] = 1.0
    weights[1, 6, 10] = 1.0
    weights[0, 7, 11] = 1.0
    weights[1, 7, 12] = 1.0
    
    weights[0, 8, 9] = 1.0
    weights[1, 8, 13] = 1.0
    weights[0, 9, 9] = 1.0
    weights[1, 9, 13] = 1.0
    weights[::, 10:-1, 0] = 1.0
    weights[0, -1, 14] = 1.0
    weights[1, -1, 15] = 1.0
    
    flatpack = weights.flatten()
    return obj(flatpack), weights


flatpack = np.random.random(2*n*(n+6))
#t0 = time.time(); obj(flatpack0); t1 = time.time(); grad(obj)(flatpack0); t2 = time.time()
res = minimize(obj, flatpack, jac=grad(obj), method='CG')

#x_data = (np.random.random((4, N)) < 0.5)
for i in range(10):
    x_data = (np.random.random((4, N)) < 0.5)
    z_data = target_fcn_1(x_data)
    full_data = np.array([x_line for x_line in x_data] + [np.ones(N), np.zeros(N)])
    res = minimize(obj, flatpack, jac=grad(obj), method='CG')
    flatpack = res['x'];
    print(res['fun'],)
    
    x_data = (np.random.random((4, N)) < 0.5)
    z_data = target_fcn_1(x_data)
    full_data = np.array([x_line for x_line in x_data] + [np.ones(N), np.zeros(N)])
    res = minimize(obj, flatpack, jac=grad(obj), method='CG')
    flatpack = res['x']
    print(res['fun'])

print(res)
print(res['fun'], res['message'])

raw_weights = np.tril(np.reshape(flatpack, (2, n, n+6)), k=5)**2
a_weights = (raw_weights[0].T/np.sum(raw_weights[0], 1)).T
b_weights = (raw_weights[1].T/np.sum(raw_weights[1], 1)).T

#    return weights

#pyplot.matshow(a_weights); pyplot.matshow(b_weights); pyplot.colorbar()
