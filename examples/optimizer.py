from ml.optimizer import GradientDescentOptimizer
import numpy as np
import matplotlib.pyplot as plt

# Data
x1 = [0.00, 4.48, 8.96, 13.44, 17.92, 22.41, 26.89, 31.37, 35.85, 40.33, 44.81]
y1 = [0.00, 2.89, 5.14, 6.74, 7.71, 8.03, 7.71, 6.74, 5.14, 2.89, 0.00]


# Defining Cost Function to optimize
def cost_function(a, b, c):
    j = 1 / len(x1)
    sigma = 0
    for i in range(len(x1)):
        sigma = sigma + (y1[i] - func(a, b, c, x1[i])) ** 2
    j = j * sigma
    return j


# Function to fit the points
def func(a, b, c, x):
    return a * (x ** 2) + b * x + c


# Define the function
# This can be compressed to:
# theta = GradientDescentOptimizer(cost_function).optimize(learning_rate=10e-7, steps=1e+5)
g = GradientDescentOptimizer(cost_function)
theta = g.optimize(learning_rate=10e-7, steps=1e+5)

#

