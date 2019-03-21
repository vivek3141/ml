import numpy as np
from ml.optimizer import GradientDescentOptimizer
from ml.graph import graph_function_and_data
from ml.error import Error
import matplotlib.pyplot as plt
from math import *
from inspect import signature


class Regression:
    def __init__(self, func, loss_function='MSE'):
        """
        Creates a new class for nonlinear regression. Function can be defined using operations on tensors
        Inbuilt functions are pow, log, exp and trig functions
        :param func: Function to fit, with the parameters in the front. Eg: m,b,x: m*x + b
        :param loss_function: Cost function to use -> examples MSE
        """
        if callable(func):
            self.func = func
            self.k = len(signature(func).parameters) - 1
        else:
            try:
                self.func = eval("lambda " + func)
                self.k = len(func.split(":")[0].split(",")) - 1
            except SyntaxError:
                raise Error("Invalid syntax for nonlinear function")

        self.theta = None
        self.x = None
        self.y = None
        self.yy = None
        self.J = None
        self.optim = None
        self.s = None
        self.x_data = None
        self.m = None

    def fit(self, x, y, init_theta=None, lr=0.001, steps=1000,
                 graph=False, dx=0.0001):
        """
        Fit the model
        :param x: X data
        :param y: Y data
        :param init_theta: Initial Theta values; can be set to none for all 0s
        :param lr: Learning Rate
        :param steps: Number of steps do go through
        :param graph: Set true to graph function and data
        :param to_print: Print loss and step number every this number
        :param batch_size: Size of batches
        :return: None
        """
        self._check_length(x, y)
        self.x = list(x)
        self.y = list(x)

        func = self.func

        if init_theta is None:
            init_theta = [0 for _ in range(self.k)]

        try:
            _ = self.func(*init_theta, 0)
        except TypeError:
            raise Error("Initial Theta does not match with function")

        def _loss(*args):
            loss = 0
            for i in range(len(x)):
                loss += (func(*args, x[i]) - y[i]) ** 2
            loss = loss * (1 / len(x))
            return loss

        optim = GradientDescentOptimizer(_loss, num_theta=self.k)
        theta = optim.optimize(
            learning_rate=lr, steps=steps, init_theta=init_theta)

        if graph:
            graph_function_and_data(lambda x: func(
                *theta, x), x_data=x, y_data=y)
        return theta

    """
    def fit(self, x, y, init_theta=None, lr=0.001, steps=1000,
            graph=False, to_print=None, batch_size=10):
        
        Fit the model
        :param x: X data
        :param y: Y data
        :param init_theta: Initial Theta values; can be set to none for all 0s
        :param lr: Learning Rate
        :param steps: Number of steps do go through
        :param graph: Set true to graph function and data
        :param to_print: Print loss and step number every this number
        :param batch_size: Size of batches
        :return: None
        
        self._check_length(x, y)

        x = np.array(x)
        y = np.array(y)

        if init_theta is None:
            init_theta = [0 for _ in range(self.k)]

        self.m = len(x)

        try:
            _ = self.func(*init_theta, 0)
        except TypeError:
            raise Error("Initial Theta does not match with function")

        # Define the graph
        self.x = tf.placeholder(tf.float32, shape=[None, 1])
        self.yy = tf.placeholder(tf.float32, shape=[None, 1])

        self.theta = [tf.Variable(initial_value=i, dtype=tf.float32)
                      for i in init_theta]

        self.y = self.func(*self.theta, self.x)
        self.J = tf.reduce_mean(tf.losses.mean_squared_error(
            labels=self.yy, predictions=self.y))

        self.optim = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.J)
        self.s = tf.Session()
        self.s.run(tf.global_variables_initializer())

        for i in range(steps):

            # Batches
            ind = np.random.choice(len(x), size=batch_size)
            x_data = np.array([x[ind]]).T
            y_data = np.matrix(y[ind]).T

            if to_print is not None and i % to_print == 0: \
                    # Mean Squared Error
                loss = sum(
                    [self.s.run(self.J, feed_dict={self.x: [[x[n]]], self.yy: [[y[n]]]}) for n in range(self.m)]) / (
                    2 * self.m)
                print(f"Step: {i}, Loss:{loss}")

            self.s.run(self.optim, feed_dict={self.x: x_data, self.yy: y_data})

        if graph:
            x1 = np.linspace(min(x), max(x), 300)
            y1 = list(map(lambda z: self.s.run(
                self.y, feed_dict={self.x: [[z]]})[0][0], x1))
            plt.scatter(x, y)
            plt.plot(x1, y1, c="r")
            plt.show()
    """

    @staticmethod
    def _check_length(x, y):
        """
        Checks the length of two arrays, if not raises ml.error.error.Error
        :param x: array 1
        :param y: array 2
        :return: None
        """
        if not (len(x) == len(y)):
            raise Error("X and Y should be the same length!")

    def predict(self, x_data):
        """
        Predict using the trained model
        :param x_data: X data to input
        :return: Predicted Value
        """
        return self.s.run(self.y, feed_dict={self.x: [[x_data]]})[0][0]

    def save(self, file_name):
        """
        Save the model
        :param file_name: File to save to model to
        :return: None
        """
        saver = tf.train.Saver()
        saver.save(self.s, str(file_name))
        self.s.close()

    def load(self, file_name):
        """
        Load a saved model
        :param file_name: File to load the model from
        :return: None
        """
        load = tf.train.Saver()
        self.s = tf.Session()
        load.restore(self.s, file_name)
