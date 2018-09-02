import numpy as np
from ml.linear_regression import LinearRegression

# Randomly generating the data
x = np.array(list(map(int, 10*np.random.random(50))))
y = np.array(list(map(int, 10*np.random.random(50))))
l = LinearRegression()
l.fit(data=x, labels=y, graph=True)
