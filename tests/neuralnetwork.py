from ml.nn import NeuralNetwork
from ml.activation import relu

n = NeuralNetwork(
    layers=[20],
    inp=784,
    activation=relu,
    out=10,
)

n.fit(51, data=mnist.train.images, labels=mnist.train.labels, lr=0.0001, graph=True)
n.test(mnist.test.images, mnist.test.labels)

