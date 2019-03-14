import matplotlib.pyplot as plt
import numpy as np


def graph_function(func, min_x, max_x, num=300):
    x_ = np.linspace(min_x, max_x, num)
    y_ = [func(i) for i in x_]

    plt.plot(x_, y_)
    plt.show()


def graph_function_and_data(func, x_data, y_data, num=300, min_x=None, max_x=None):
    min_x = min(x_data) if min_x is None else min_x
    max_x = max(x_data) if max_x is None else max_x

    x_ = np.linspace(min_x, max_x, num)
    y_ = [func(i) for i in x_]

    plt.plot(x_, y_)
    plt.scatter(x_data, y_data, color="RED")
    plt.show()
