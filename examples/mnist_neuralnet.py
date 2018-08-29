from ml.nn import NeuralNetwork
from ml.activation import sigmoid
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

n = NeuralNetwork(
    layers=20,
    inp=784,
    activation=sigmoid,
    out=10,
)

# n.load("model.pkl")

n.fit(10000, data=mnist.train.images, labels=mnist.train.labels, lr=0.0001, graph=True)

n.test(mnist.test.images, mnist.test.labels)

n.save("model.pkl")
