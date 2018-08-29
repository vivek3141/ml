import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_d(x):
    return sigmoid(x) * (1 - sigmoid(x))


def tanh(x):
    return np.tanh(x)


def tanh_d(x):
    return 1 - (np.tanh(x) ** 2)


def relu(x):
    return max(0, x)
