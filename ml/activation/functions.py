import tensorflow as tf


def sigmoid(x):
    return 1 / (1 + tf.exp(-x))


def sigmoid_d(x):
    return sigmoid(x) * (1 - sigmoid(x))


def tanh(x):
    return tf.tanh(x)


def tanh_d(x):
    return 1 - (tf.tanh(x) ** 2)


def relu(x):
    return tf.nn.relu(x)
