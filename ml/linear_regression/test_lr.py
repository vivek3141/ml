import linear_regression
import numpy as np

x = list(map(int, 10*np.random.random(50)))
y = list(map(int, 10*np.random.random(50)))
theta = linear_regression.fit(x, y, 0.0001, 1000, [1,1])