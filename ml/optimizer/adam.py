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
        steps=10000,
        init_theta=None,
        dx=0.0001
    ):
        theta_0 = [0 for i in range(self.num_theta)
                   ] if init_theta is None else init_theta
        m_t = 0
        v_t = 0
        for i in range(steps):
            g_t = self.gradient(theta_0)
            m_t = beta1*m_t + (1-beta1)*g_t
            v_t = beta2*v_t + (1-beta2)*(g_t*g_t)
            m_cap = m_t/(1-(beta1**i))
            v_cap = v_t/(1-(beta2**i))
            theta_0_prev = theta_0
            theta_0 = theta_0 - (alpha*m_cap)/(math.sqrt(v_cap) +
                                               epsilon)
            if(theta_0 == theta_0_prev):
                break

    def gradient(self, theta, dx=0.0001):
        partials = []
        for t in range(self.num_theta):
            theta_dx = [(theta[x] + dx) if t == x else theta[x]
                        for x in range(self.num_theta)]
            partial = (self.func(*theta_dx) - self.func(*theta)) / dx
            partials.append(partial)
        return np.array(partials)


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


# Optimize
g = AdamOptimizer(cost_function)
theta = g._optimize_python()

# Graph the function by using the theta learnt
graph_function_and_data(lambda x: func(*theta, x), x_data=x1, y_data=y1)
