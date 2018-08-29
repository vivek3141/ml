import numpy as np
from ml.logistic_regression import LogisticRegression
import matplotlib.pyplot as plt
np.random.seed(5)
x1 = np.array(list(map(lambda z: int(z), 100 * np.random.random(100))))
x2 = np.array(list(map(lambda z: int(z), 100 * np.random.random(100))))
y = np.array([1 if x1[i] > x2[i] else 0 for i in range(100)])
l = LogisticRegression(features=2)
data = [[x1[i], x2[i]] for i in range(100)]
l.fit(data=data, labels=x2, graph=True, lr=0.01)
for i in range(100):
    if bool(y[i]):
        plt.scatter(x1[i], x2[i], c="RED")
    else:
        plt.scatter(x1[i], x2[i], c="BLUE")
plt.show()
l.test(data=data, labels=y)
