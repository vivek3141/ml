import tensorflow as tf

class CNN2D():
    def __init__(self, layers, inp, out, kernel_size, activation, pool_size):
        self.input = inp
        self.output = out
        self.layers = layers
        self.conv = None
        self.pool = None
        self.kernel_size = kernel_size
        self.activation = activation
        self.pool_size = pool_size

    def create_model(self, data, labels, mode):
        size = [-1, len(data), len(data[0]), 1]
        inp = tf.reshape(data, size)
        self.conv = tf.layers.conv2d(
            inputs=inp,
            filters=32,
            kernel_size=self.kernel_size,
            padding="same",
            activation=self.activation,
        )
        self.pool = tf.layers.max_pooling2d(inputs=self.conv, pool_size=self.pool_size, strides=2)
        for i in range(1, len(self.layers)-1):
            pass



