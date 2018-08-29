try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    print("Warning: Can't graph from terminal")
import numpy as np


class LinearRegression:
    def __init__(self):
        self.theta = []

    def _linear_regression(self, x, label, m=0, b=0, steps=1000, lr=0.0001):
        n = float(len(label))
        for i in range(steps):
            y = (m * x) + b
            cost = sum([data ** 2 for data in (label - y)]) / n
            m_gradient = -(2 / n) * sum(x * (label - y))
            b_gradient = -(2 / n) * sum(label - y)
            m = m - (lr * m_gradient)
            b = b - (lr * b_gradient)
        return m, b, cost

    def _matrix_sub(self, mat1, mat2):
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

    def _hypothesis(self, theta, x):
        return theta[0] * x + theta[1]

    def fit(self, data, labels, lr=0.001, graph=False, steps=1000, init_theta=[1, 1]):
        if steps > len(data):
            steps = len(data)
        theta = init_theta
        x = data
        y = labels
        n = len(x)
        for i in range(len(x)):
            theta = self._matrix_sub(theta, self._gradient(x[i], y[i], theta, n))

        m, b, cost = self._linear_regression(x, y, steps=steps, lr=lr)
        theta[0] = m
        theta[1] = b
        self.theta = theta
        xLine = np.array(range(0, 10))
        yLine = theta[0] * xLine + theta[1]
        if graph:
            plt.scatter(x, y, c="RED")
            plt.plot(xLine, yLine)
            plt.show()
        return cost

    def predict(self, x):
        return self._hypothesis(self.theta, x)

    def save(self, file_name):
        with open(file_name, "w") as f:
            f.write(str(self.theta[0]) + " " + str(self.theta[1]))
        print("Saved to: {}".format(file_name))

    def load(self, file_name):
        with open(file_name, "r") as f:
            s = f.read().split(" ")
        self.theta[0] = int(s[0])
        self.theta[1] = int(s[0])
