import tensorflow as tf
from ml.activation import relu


class CNN2D:
    def __init__(self, layers, inp, out, kernel_size, pool_size, filters, activation=relu, strides=2, drop_rate=0.5):
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
        self.drop = drop_rate
        self.lr = 0

    def _create_model(self, data, labels, mode):
        size = [-1, len(data), len(data[0]), 1]
        train = mode == tf.estimator.ModeKeys.TRAIN
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
        units = int(tf.size(self.pool)) * self.filters[-1]
        pool_flat = tf.reshape(self.pool, [-1, units])
        dense = tf.layers.dense(inputs=pool_flat, units=units, activation=self.activation)
        drop = tf.layers.dropout(inputs=dense, rate=self.drop, training=train)
        logits = tf.layers.dense(inputs=drop, units=self.output)
        p = {"classes": tf.argmax(logits, axis=1), "probabilities": tf.nn.softmax(logits, name="softmax_tensor")}
        if not train:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=p)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        if train:
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=op)
        eval = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=p["classes"])}
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval)

    def fit(self, data, labels, lr):
        self.lr = lr
