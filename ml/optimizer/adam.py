from ml.graph import graph_function_and_data
from inspect import signature
import numpy as np
import math


class AdamOptimizer:
    def __init__(self, func, num_theta=None):
        self.func = func
        self.num_theta = len(
            signature(func).parameters) if num_theta is None else num_theta

    def _optimize_python(
        self,
        alpha=0.01,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        steps=1,
        init_theta=None,
        dx=0.0001
    ):
        theta = np.array([0 for i in range(self.num_theta)
                          ] if init_theta is None else init_theta)
        m = 0
        v = 0
        for i in range(steps):
            grad = self.gradient(theta)
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)
            m_h = m / (1 - (beta1 ** i))
            v_h = m / (1 - (beta2 ** i))
            theta = theta - alpha * (m_h / (np.sqrt(v_h) - epsilon))
        return theta

    def gradient(self, theta, dx=0.001):
        partials = []
        for t in range(self.num_theta):
            theta_dx = [(theta[x] + dx) if t == x else theta[x]
                        for x in range(self.num_theta)]
            partial = (self.func(*theta_dx) - self.func(*theta)) / dx
            partials.append(partial)
