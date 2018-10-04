import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib import colors as mcolors
import random


def plot(data, assignments, centers):
    li = ["r", "g", "b", "y"]
    for k, i in enumerate(data):
        plt.scatter(i[0], i[1], color=li[assignments[k]])
    plt.scatter(centers[:][0], centers[:][1], color="black")
    plt.show()
