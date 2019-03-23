import linear_regression
import numpy as np
import matplotlib.pyplot as plt

x = list(map(int, 10*np.random.random(50)))
y = list(map(int, 10*np.random.random(50)))

theta = linear_regression.fit(x, y, 0.01, 1000, [1, 1], len(x))

x_line = np.array(range(0, 10))
y_line = theta[0] * x_line + theta[1]
if graph:
    plt.scatter(x, y, c="RED")
    plt.plot(x_line, y_line)
    plt.show()

