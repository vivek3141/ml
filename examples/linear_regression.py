import numpy as np
from ml.linear_regression import LinearRegression

# Randomly generating the data
x = np.array(list(map(lambda z: int(z), 10*np.random.random(25))))
y = np.array(list(map(lambda z: int(z), 10*np.random.random(25))))
l = LinearRegression()
l.fit(data=x, labels=y, graph=True)