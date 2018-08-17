from ml.nn import NeuralNetwork
from ml.activation import sigmoid
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

n = NeuralNetwork(
    layers=20,
    data=mnist.train.images,
    activation=sigmoid,
    labels=mnist.train.labels,
    lr=0.0001
)
n.fit(10000, graph=True)
