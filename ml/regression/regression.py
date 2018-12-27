import numpy as np
from ml.error import Error
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import pow, log, exp


class Regression:
    def __init__(self, func):
        self.func = eval("lambda " + func)
        self.theta = None
        self.x = None
        self.y = None
        self.yy = None
        self.J = None
        self.optim = None
        self.s = None
        self.x_data = None
        self.m = None

    def tf_func(self, theta):
        t = []
        for i in theta:
            t.append(self.s.run(i))
        return

    def fit(self, x, y, init_theta, lr=0.001, steps=1000, graph=False, to_print=None):
        self.check_length(x, y)
        k = 3
        self.m = len(x)
        try:
            _ = self.func(*[0 for _ in range(k)], 0)
        except TypeError:
            raise Error("Initial Theta does not match with function")
        self.x = tf.placeholder(tf.float32, shape=[None, 1])
        self.yy = tf.placeholder(tf.float32, shape=[None, 1])
        self.theta = [tf.Variable(initial_value=i, dtype=tf.float32) for i in init_theta]
        self.y = self.func(*self.theta, self.x)
        self.J = tf.reduce_mean(tf.losses.mean_squared_error(labels=self.yy, predictions=self.y))
        self.optim = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.J)
        self.s = tf.Session()
        self.s.run(tf.global_variables_initializer())
        loss = []
        for i in range(steps):
            x_data = [[x[i % self.m]]]
            y_data = [[y[i % self.m]]]
            if to_print is not None and i % to_print == 0:
                loss.append(
                    sum([self.s.run(self.J, feed_dict={self.x: [[x[n]]], self.yy: [[y[n]]]}) for n in
                         range(self.m)]) / (2 * self.m))
                print(f"Step: {i}, Loss:{loss[-1]}")
            self.s.run(self.optim, feed_dict={self.x: x_data, self.yy: y_data})
        if graph:
            x1 = np.linspace(min(x), max(x), 300)
            y1 = list(map(lambda z: self.s.run(self.y, feed_dict={self.x: [[z]]})[0][0], x1))
            plt.scatter(x, y)
            plt.plot(x1, y1, c="r")
            plt.show()

    @staticmethod
    def subtract(li1, li2):
        Regression.check_length(li1, li2)
        ret = []
        for i in range(len(li1)):
            ret.append(li1[i] - li2[i])
        return ret

    @staticmethod
    def check_length(x, y):
        if not (len(x) == len(y)):
            raise Error("X and Y should be the same length!")

    def cost(self, x, y, theta):
        self.check_length(x, y)
        cost = 0
        for i in range(len(x)):
            cost += np.power(self.func(*theta, x[i]) - y[i], 2)
        return cost

    def predict(self):
        pass

    def save(self):
        pass

    def load(self):
        pass
