import numpy as np
from inspect import signature


class GradientDescentOptimizer:
    def __init__(self, func):
        self.func = func
        self.num_theta = len(signature(func).parameters)
        print(self.num_theta)

    def optimize(self, learning_rate=0.01, steps=1000, init_theta=None, dx=0.0001):
        theta = [0 for _ in range(self.num_theta)] if init_theta is None else init_theta
        for i in range(steps):
            t_grad = []
            for t in range(self.num_theta):
                t_grad.append(learning_rate *
                              ((self.func(*[theta[x] + dx if t == x else theta[x] for x in range(self.num_theta)])
                                - self.func(*[theta])) / dx)
                              )
            for k in range(self.num_theta):
                theta[k] -= t_grad[k]
        return theta
