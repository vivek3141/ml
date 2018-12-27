from ml.regression import Regression
import tensorflow as tf

x1 = [0.00, 4.48, 8.96, 13.44, 17.92, 22.41, 26.89, 31.37, 35.85, 40.33, 44.81]
y1 = [0.00, 2.89, 5.14, 6.74, 7.71, 8.03, 7.71, 6.74, 5.14, 2.89, 0.00]


def func(a, b, c, x):
    return tf.multiply(a, tf.pow(x, 2)) + tf.multiply(b, x) + c


r = Regression(func)
t = r.fit(x1, y1, [1, 1, 1], graph=True, to_print=50)
