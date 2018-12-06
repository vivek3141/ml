import numpy as np
from ml.linear_regression import LinearRegression

x = np.array(list(map(int, np.random.random(2))))
y = np.array(list(map(int, np.random.random(2))))
l = LinearRegression()
l.fit(data=x, labels=y, graph=False)
