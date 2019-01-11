from ml.optimizer import GradientDescentOptimizer
import numpy as np

x = np.array(list(map(int, 10 * np.random.random(50))))
y = np.array(list(map(int, 10 * np.random.random(50))))


def cost_function(theta_0, theta_1):
    j = 1 / len(x)
    sigma = 0
    for i in range(len(x)):
        sigma = sigma + (y[i] - (theta_0 + theta_1 * x[i])) ** 2
    j = j * sigma
    return j


g = GradientDescentOptimizer(cost_function)
