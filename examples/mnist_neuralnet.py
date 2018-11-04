from ml.nn import NeuralNetwork
from ml.activation import relu
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

n = NeuralNetwork(
    layers=[20],
    inp=784,
    activation=relu,
    out=10,
)

# n.load("model.pkl")

n.fit(50000, data=mnist.train.images, labels=mnist.train.labels, lr=0.0001, graph=True)

n.test(mnist.test.images, mnist.test.labels)  # 91.52%, can be better!

# n.save("model.pkl")
