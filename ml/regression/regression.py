import numpy as np
from ml.error import Error
import matplotlib.pyplot as plt
import tensorflow as tf


class Regression:
    def __init__(self, func):
        self.func = func
        self.theta = None
        self.x = None
        self.y = None
        self.yy = None
        self.J = None
        self.optim = None
        self.s = None
        self.x_data = None

    def tf_func(self, theta):
        t = []
        for i in theta:
            t.append(self.s.run(i))
        return 

    def fit(self, x, y, init_theta, steps=1000, lr=0.01, graph=False, to_print=None):
        self.check_length(x, y)
        try:
            ret = self.func(*init_theta, 0)
        except TypeError:
            raise Error("Initial Theta does not match with function")
        self.x = tf.placeholder(tf.float32, shape=[None, 1])
        self.yy = tf.placeholder(tf.float32, shape=[None, 1])
        self.theta = np.array([tf.get_variable(f"T{i}", [1, 1]) for i in range(len(init_theta))])
        self.s = tf.Session()
        self.s.run(tf.global_variables_initializer())
        # self.y = tf.Variable(self.tf_func(self.theta))
        self.y = self.func(self.theta, self.x)
        self.J = tf.reduce_mean(tf.losses.mean_squared_error(labels=self.yy, predictions=self.y))
        self.optim = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.J)
        loss = []
        for i in range(steps):
            x_data = x[i]
            print(x_data)
            y_data = y[i]
            if to_print is not None and i % to_print == 0:
                loss.append(
                    self.s.run(self.J, feed_dict={self.x: [[x[0]]], self.yy: [[y[0]]]}))
                print(f"Step: {i}, Loss:{loss[-1]}")
            self.s.run(self.optim, feed_dict={self.x: x_data, self.yy: y_data})
            """
            for i in range(steps):
                j = self.cost(x, y, theta)
                dx = 0.0001
                t_grad = [0 for x in range(len(theta))]

                for k, n in enumerate(theta):
                    t = theta[:]
                    t[k] = t[k] + dx
                    t_grad[k] = lr * ((self.cost(x, y, t) - j) / dx)
                theta = self.subtract(theta, t_grad)
                print(t_grad, theta)
                """
        if graph:
            x1 = np.linspace(min(x), max(x), 300)
            y1 = list(map(lambda z: self.func(*self.theta, z), x1))
            plt.scatter(x, y)
            plt.plot(x1, y1)
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
            Error("X and Y should be the same length!")

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
