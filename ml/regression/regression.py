import tensorflow as tf
import numpy as np
from ml.error import Error


class Regression:
    def __init__(self, func):
        self.func = func

    def fit(self, x, y, init_theta, steps=1000, lr=0.01, graph=False):
        pass

    def check_length(self, x, y):
        if not (len(x) == len(y)):
            Error("X and Y should be the same length!")

    def cost(self, x, y, theta):
        theta = np.array(theta)
        for i in range(len(x)):
            pass

    def predict(self):
        pass

    def save(self):
        pass

    def load(self):
        pass
