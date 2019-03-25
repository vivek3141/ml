import lin_reg
import numpy as np
import matplotlib.pyplot as plt
from ml.linear_regression import LinearRegression
import time

x = sorted([0.1, 2.3, 6.3, 9.2, 1.6, 2.3, 5.1, 7.2, 4.3, 1.5, 6.3, 8.9, 3.2, 3.5, 5.6])
y = list(map(lambda x: 0.3*x + 0.2, x))

start_time = time.time()
#theta = LinearRegression()
#theta.fit(np.array(x), np.array(y), lr=0.01)
#theta = theta.theta
theta = lin_reg.fit(x, y, 0.01, 1000, [1, 1], len(x))

print("--- %s seconds ---" % (time.time() - start_time))

x_line = np.array([x[0], x[-1]])
y_line = theta[0] * x_line + theta[1]

plt.scatter(x, y, c="RED")
plt.plot(x_line, y_line)
plt.show()

