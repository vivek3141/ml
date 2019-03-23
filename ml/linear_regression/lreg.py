import matplotlib.pyplot as plt
import numpy as np


class LinearRegression:
    def __init__(self):
        self.theta = []

    @staticmethod
    def _linear_regression(x, label, m=0, b=0, steps=1000, lr=0.0001):
        n = float(len(label))
        cost = 0
        for i in range(steps):
            y = (m * x) + b
            cost = sum([data ** 2 for data in (label - y)]) / n
            m_gradient = -(2 / n) * sum(x * (label - y))
            b_gradient = -(2 / n) * sum(label - y)
            m = m - (lr * m_gradient)
            b = b - (lr * b_gradient)
        return m, b, cost

    @staticmethod
    def _matrix_sub(mat1, mat2):
        mat = []
        for i in range(len(mat1)):
            mat.append(mat1[i] - mat2[i])
        return mat

    def _cost_function(self, theta, x, y):
        j = 1 / len(x)
        sigma = 0
        for i in range(len(x)):
            sigma = sigma + (y[i] - self._hypothesis(theta, x[i]))
        j = j * sigma
        return j

    def _gradient(self, x, y, theta, n):
        weight = (2 / n) * (-x) * (y - self._hypothesis(theta, x))
        bias = (2 / n) * (y - self._hypothesis(theta, x))
        return [weight, bias]

    @staticmethod
    def _hypothesis(theta, x):
        return theta[0] * x + theta[1]

    def fit(self, data, labels, lr=0.001, graph=False, steps=1000, init_theta=(1, 1)):
        """
        Fit the model with the given data
        :param data: Data input matrix
        :param labels: Data input labels
        :param lr: Learning rate
        :param graph: True if to graph the model
        :param steps: Number of steps
        :param init_theta: Initial theta values
        :return: Cost of the model
        """
        if steps > len(data):
            steps = len(data)
        theta = init_theta
        x = data
        y = labels
        n = len(x)
        for i in range(len(x)):
            theta = LinearRegression._matrix_sub(theta, self._gradient(x[i], y[i], theta, n))

        m, b, cost = LinearRegression._linear_regression(x, y, steps=steps, lr=lr)
        theta[0] = m
        theta[1] = b
        self.theta = theta
        x_line = np.array(range(0, 10))
        y_line = theta[0] * x_line + theta[1]
        if graph:
            plt.scatter(x, y, c="RED")
            plt.plot(x_line, y_line)
            plt.show()
        return cost

    def predict(self, x):
        """
        Predict a value with the model
        :param x: Input
        :return: Predicted Value
        """
        return LinearRegression._hypothesis(self.theta, x)

    def save(self, file_name):
        """
        Save the model
        :param file_name: File to save the model to
        :return: None
        """
        with open(file_name, "w") as f:
            f.write(str(self.theta[0]) + " " + str(self.theta[1]))
        print("Saved to: {}".format(file_name))

    def load(self, file_name):
        """
        Load the model from the file
        :param file_name: File to load the model from
        :return: None
        """
        with open(file_name, "r") as f:
            s = f.read().split(" ")
        self.theta[0] = int(s[0])
        self.theta[1] = int(s[0])
