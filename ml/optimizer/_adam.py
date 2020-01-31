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
        alpha=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        steps=10000,
        init_theta=None,
        dx=0.001
    ):
        theta = np.array([0 for i in range(self.num_theta)
                          ] if init_theta is None else init_theta)
        m = 0
        v = 0
        for i in range(1, steps):
            grad = self.gradient(theta, dx=dx)
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)
            m_h = m / (1 - (beta1 ** i))
            v_h = v / (1 - (beta2 ** i))
            delta = alpha * (m_h / (np.sqrt(v_h) - epsilon))
            theta_old = theta
            theta = theta - delta
            if i % 50 == 0:
                print(f"Step: {i} Cost {cost}")
        return theta

    def gradient(self, theta, dx=0.001):
        partials = []
        for t in range(self.num_theta):
            theta_dx = [(theta[x] + dx) if t == x else theta[x]
                        for x in range(self.num_theta)]
            partial = (self.func(*theta_dx) - self.func(*theta)) / dx
            partials.append(partial)
        return np.array(partials)