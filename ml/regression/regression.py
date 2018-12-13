import numpy as np
from ml.error import Error
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


class Regression:
    def __init__(self, func):
        self.func = func

    def fit(self, x, y, init_theta, steps=1000, lr=0.01, graph=False):
        self.check_length(x, y)
        theta = np.array(init_theta)
        for i in range(steps):
            J = self.cost(x, y, theta)
            dx = 0.0001
            t_grad = [0 for x in range(len(theta))]

            for k, n in enumerate(theta):
                t = theta[:]
                t[k] = t[k] + dx
                t_grad[i] = lr * ((self.cost(x, y, t) - J) / dx)
            theta = self.subtract(theta, t_grad)
        if graph:
            plt.show()
        return theta

    @staticmethod
    def subtract(li1, li2):
        Regression.check_length(li1, li2)
        ret = []
        for i in range(len(li1)):
            ret.append(li1[i] - li2[i])
        return ret

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
