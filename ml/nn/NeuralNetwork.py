import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from ml.activation import *

class NeuralNetwork:
    def __init__(self, layers, data, labels, input=208, output=5, activation='sigmoid', lr=0.001):
        act = {'sigmoid': sigmoid, 'tanh': tanh}
        derivative = {'sigmoid': sigmoid_d, 'tanh': tanh_d}
        try:
            self.activation = act[activation]
            self.activation_d = derivative[activation]
        except KeyError:
            print("Invalid Activation Function")
            self.activation = sigmoid
            self.activation_d = sigmoid_d
        self.W = []
        self.b = []
        self.y = []
        self.layers = layers
        self.input = input
        self.output = output
        self.lr = lr
        self.data = data
        self.labels = labels

    def create_network(self):
        x = tf.placeholder(tf.float32, shape=[None, self.input])
        yy = tf.placeholder(tf.float32, shape=[None, self.output])
        z = self.create_layer(0, self.input, self.layers, x, False)
        y = self.create_layer(1, self.layers, self.output, z, False)
        J = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=yy, logits=y))
        optim = tf.train.AdamOptimizer(learning_rate=self.lr)
        min = optim.minimize(J)
        return J, min, x, y, z, yy

    def fit(self):
        J, min, x, y, z, yy = self.create_network()
        s = tf.Session()
        s.run(tf.global_variables_initializer())
        losses = []
        num = 0
        for i in range(10000):
            xval = self.data[i].reshape(1, self.input)
            yval = self.labels[i].reshape(1, self.output)
            if i % 50:
                losses.append(s.run(J, feed_dict={x: self.data[500].reshape(1, self.input),
                                                  yy: self.labels[500].reshape(1, self.output)}))
            a = s.run(min, feed_dict={x: xval, yy: yval})

        plt.plot(range(len(losses)), losses)
        plt.show()

    def create_layer(self, layer_number, input_size, output_size, x, nonlinear=True):
        with tf.variable_scope(f"layer_{layer_number}"):
            self.W = tf.get_variable("W", [input_size, output_size])
            self.b = tf.get_variable("b", [1, output_size])
            y = tf.matmul(x, self.W) + self.b
            if nonlinear:
                y = self.activation(self.y)
            return y


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

s = NeuralNetwork(
    layers=20,
    data=mnist.train.images,
    labels=mnist.train.labels,
    input=784,
    output=10
)
s.fit()
