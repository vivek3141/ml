from ml.logistic_regression import LogisticRegression
from ml.random import yx2features

data, labels = yx2features(number=3)

lr = LogisticRegression(features=2)
lr.fit(data=data, labels=labels, graph=True, lr=0.01, steps=200, to_print=False)

lr.test(data=data, labels=labels)
