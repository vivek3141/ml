import tensorflow as tf
import numpy as np
from ml.error import Error
import matplotlib.pyplot as plt


class Regression:
    def __init__(self, func):
        self.func = func

    def fit(self, x, y, init_theta, steps=1000, lr=0.01, graph=False):
        self.check_length(x, y)
        theta = np.array(init_theta)
        for i in range(steps):
            J = self.cost(x, y, theta)
            j_grad =
        if graph:
            plt.show()
        pass

    @staticmethod
    def check_length(x, y):
        if not (len(x) == len(y)):
            Error("X and Y should be the same length!")

    def cost(self, x, y, theta):
        self.check_length(x, y)
        cost = 0
        for i in range(len(x)):
            cost += np.power(self.func(*theta, x[i]) - y[i], 2)
        return cost

    def predict(self):
        pass

    def save(self):
        pass

    def load(self):
        pass
