import tensorflow as tf
from ml.activation import relu
import numpy as np
import pickle


class CNN2D:
    def __init__(self, layers, inp, out, kernel_size, pool_size, filters, dimensions, activation=relu, strides=2,
                 drop_rate=0.5):
        self.input = inp
        self.output = out
        self.layers = layers
        self.conv = None
        self.test_data = None
        self.test_labels = None
        self.pool = None
        self.data = None
        self.labels = None
        self.epochs = None
        self.kernel_size = kernel_size
        self.activation = activation
        self.filters = filters
        self.strides = strides
        self.pool_size = pool_size
        self.drop = drop_rate
        self.lr = 0
        self.classifier = None
        self.dimensions = dimensions

    def _create_model(self, features, labels, mode):
        data = features
        size = [-1, self.dimensions[0], self.dimensions[1], 1]
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

        units = int(self.dimensions[0] * self.dimensions[1] / (self.strides ** (2 * self.layers))) * self.filters[-1]

        pool_flat = tf.reshape(self.pool, [-1, units])
        dense = tf.layers.dense(inputs=pool_flat, units=1024, activation=self.activation)
        drop = tf.layers.dropout(inputs=dense, rate=self.drop, training=train)
        logits = tf.layers.dense(inputs=drop, units=self.output)
        p = {"classes": tf.argmax(logits, axis=1), "probabilities": tf.nn.softmax(logits, name="softmax_tensor")}
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        if not train:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=p, loss=loss)

        if train:
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=op)

        eval = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=p["classes"])}
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval)

    def _fit(self, data, labels, epochs, to_print, save_path):

        self.classifier = tf.estimator.Estimator(
            model_fn=self._create_model, model_dir=save_path)

        tensors_to_log = {"probabilities": "softmax_tensor"}
        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=50)

        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x=data,
            y=labels,
            batch_size=100,
            num_epochs=None,
            shuffle=True)

        self.classifier.train(
            input_fn=train_input_fn,
            steps=epochs,
            hooks=[logging_hook])
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x=data,
            y=labels,
            num_epochs=1,
            shuffle=False)

        eval_results = self.classifier.evaluate(input_fn=eval_input_fn)
        if to_print:
            print("Done Training")
            [print(str(i) + " : " + str(eval_results[i])) for i in eval_results.keys()]
        print("")
        self.eval_results = eval_results

    def predict(self, data):
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x=data,
            shuffle=False)
        eval_results = self.classifier.evaluate(input_fn=eval_input_fn)
        return next(eval_results)

    def fit(self, data, labels, lr, epochs, to_print=False, save_path="./CNN_model"):
        self.lr = lr
        self._fit(data, labels, epochs, to_print, save_path)
        return self.eval_results

    def _test(self, data, labels, to_print=False):
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x=data,
            y=labels,
            num_epochs=1,
            shuffle=False)

        eval_results = self.classifier.evaluate(input_fn=eval_input_fn)

        if to_print:
            print("Done Testing")
            [print(str(i) + " : " + str(eval_results[i])) for i in eval_results.keys()]
        print("")
        self.eval_results = eval_results

    def test(self, data, labels, to_print=False):
        self._test(data, labels, to_print)
        return self.eval_results

    def save(self, file_name):
        s = tf.train.Saver()
        path = s.save(self.classifier, "./" + file_name)
        #with open(file_name, "wb") as w:
        #    pickle.dump(self.classifier, w, pickle.HIGHEST_PROTOCOL)
        print("Saved to " + path)

    def load(self, file_name):
        s = tf.train.Saver()
        s.restore(self.classifier, "./" + file_name)
        #with open(file_name, "rb") as w:
        #    self.classifier = pickle.load(w)
