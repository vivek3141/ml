import matplotlib.pyplot as plt
import numpy as np


def scatter_two_features(data, labels):
    [plt.scatter(data[i][0], data[i][1], c="RED")
     if bool(labels[i]) else plt.scatter(data[i][0], data[i][1], c="BLUE") for i in range(100)]


def plot_line(w, b):
    li = np.arange(0, 100, 2)
    plt.plot(li, np.vectorize(lambda x: 0.5 - w[0] / w[1] * x - b / w[1])(li))
