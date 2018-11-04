import numpy as np
from ml.error import Error
import matplotlib.pyplot as plt


def yx2features(ran=[0, 100], number=100):
    if not len(ran) == 2:
        raise Error("Range must be a list of 2 numbers!")
    func = lambda: ((max(ran) - min(ran) + 1) * np.random.random(number)) + min(ran)
    x1 = np.array(list(map(lambda z: int(z), func())))
    x2 = np.array(list(map(lambda z: int(z), func())))
    y = np.array([1 if x1[i] > x2[i] else 0 for i in range(number)])
    data = [[x1[i], x2[i]] for i in range(number)]

    return data, y
