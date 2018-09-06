import tensorflow as tf
from ml.activation import relu


class CNN2D:
    def __init__(self, layers, inp, out, kernel_size, pool_size, filters, activation=relu, strides=2):
        self.input = inp
        self.output = out
        self.layers = layers
        self.conv = None
        self.pool = None
        self.kernel_size = kernel_size
        self.activation = activation
        self.filters = filters
        self.strides = strides
        self.pool_size = pool_size

    def _create_model(self, data, labels, mode):
        size = [-1, len(data), len(data[0]), 1]
        inp = tf.reshape(data, size)
        self.conv = tf.layers.conv2d(
            inputs=inp,
            filters=self.filters[0],
            kernel_size=self.kernel_size,
            padding="same",
            activation=self.activation,
        )
        self.pool = tf.layers.max_pooling2d(inputs=self.conv, pool_size=self.pool_size, strides=self.strides)
        for i in range(1, self.layers):
            self.conv = tf.layers.conv2d(
                inputs=self.pool,
                filters=self.filters[i],
                kernel_size=self.kernel_size,
                padding="same",
                activation=self.activation
            )
            self.pool = tf.layers.max_pooling2d(inputs=self.conv, pool_size=self.pool_size, strides=self.strides)
        pool_flat = tf.reshape(self.pool, [-1, int(tf.size(self.pool))])
        dense = tf.layers.dense(inputs=pool_flat, units=self.output)

