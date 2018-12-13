from ml.regression import Regression

x1 = [0.00, 4.48, 8.96, 13.44, 17.92, 22.41, 26.89, 31.37, 35.85, 40.33, 44.81]
y1 = [0.00, 2.89, 5.14, 6.74, 7.71, 8.03, 7.71, 6.74, 5.14, 2.89, 0.00]


def func(a, b, c, x):
    return a * x ** 2 + b * x + c


r = Regression(func)
theta = r.fit(x1, y1, [1, 1, 1], graph=True)
