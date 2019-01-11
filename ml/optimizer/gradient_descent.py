import numpy as np
from inspect import signature


class GradientDescentOptimizer:
    def __init__(self, func):
        self.func = func
        self.num_theta = len(signature(func).parameters)
        print(self.num_theta)

    def optimize(self, learning_rate, steps, init_theta=None):
        theta = [0 for i in range(self.num_theta)] if init_theta is None
        for i in range(steps):
            for i in range(self.num_theta):


