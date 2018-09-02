from ml.logistic_regression import LogisticRegression
import matplotlib.pyplot as plt
from ml.random import yx2features
from ml.graph.lr import scatter_two_features, plot_line

data, labels = yx2features()

lr = LogisticRegression(features=2)
lr.fit(data=data, labels=labels, graph=True, lr=0.01, steps=200)

w, b = lr.get_values()

scatter_two_features(data, labels)
plot_line(w, b)
plt.show()

lr.test(data=data, labels=labels)
