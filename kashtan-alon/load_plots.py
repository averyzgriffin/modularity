from matplotlib import pyplot as plt
import pickle

file_path = "loss_curves/loss_mvg20_gensize500_pm0.1.pickle"

with open(file_path, 'rb') as fib:
    fig = pickle.load(fib)

plt.show()

