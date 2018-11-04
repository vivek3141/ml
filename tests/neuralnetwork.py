from ml.nn import NeuralNetwork
from ml.activation import relu
import pickle
import numpy as np

with open("label.txt", "rb") as f:
    label = pickle.load(f)
with open("data.txt", "rb") as f:
    image = pickle.load(f)

n = NeuralNetwork(
    layers=[20],
    inp=784,
    activation=relu,
    out=10,
)

n.fit(1, data=np.array([image, image]), labels=np.array([[0] * 10, [0] * 10]), lr=0.001)
n.test(np.array([image]), np.array([label], dtype=np.int32))
