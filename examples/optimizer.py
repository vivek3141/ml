from ml.optimizer import GradientDescentOptimizer
import numpy as np
import matplotlib.pyplot as plt

x = [0.00, 4.48, 8.96, 13.44, 17.92, 22.41, 26.89, 31.37, 35.85, 40.33, 44.81]
y = [0.00, 2.89, 5.14, 6.74, 7.71, 8.03, 7.71, 6.74, 5.14, 2.89, 0.00]


def cost_function(a, b, c):
    j = 1 / len(x)
    sigma = 0
    for i in range(len(x)):
        sigma = sigma + (y[i] - func(a, b, c, x[i])) ** 2
    j = j * sigma
    return j


def func(a, b, c, x):
    return a * (x ** 2) + b * x + c


g = GradientDescentOptimizer(cost_function)
theta = g.optimize(learning_rate=10e-7, steps=1e+5)

x_ = np.linspace(min(x), max(x), 300)
y_ = [func(*theta, i) for i in x_]

plt.plot(x_, y_)
plt.scatter(x, y)

plt.show()
