import tensorflow as tf
import numpy as np
from ml.activation import sigmoid

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    print("Warning: Can't graph from terminal")


class LogisticRegression:
    def __init__(self, features, seed=5):
        self.input = features
        tf.set_random_seed(seed=seed)
        self.W = tf.Variable(tf.random_normal(shape=[self.input, 1]))
        self.b = tf.Variable(tf.random_normal(shape=[1, 1]))
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.input])
        self.yy = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.s = tf.Session()
        self.y = tf.matmul(self.x, self.W) + self.b
        self.J = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.yy, logits=self.y))

    def test(self, data, labels):
        correct = 0
        for i in range(len(data)):
            if labels[i] == self.predict(data[i]):
                correct += 1
        print("Accuracy: {}%".format(correct / len(data) * 100))

    def predict(self, data, to_round=True):
        if to_round:
            return round((self.s.run(self.y, feed_dict={self.x: np.matrix(data)})[0][0]))
        return self.s.run(self.y, feed_dict={self.x: np.matrix(data)})[0][0]

    def fit(self, data, labels, lr=0.001, steps=1000, graph=False, batch_size=25):
        minimize = tf.train.AdamOptimizer(learning_rate=lr)
        optimize = minimize.minimize(self.J)
        pred = tf.round(tf.sigmoid(self.y))
        is_correct = tf.cast(tf.equal(pred, self.yy), dtype=tf.float32)
        accuracy = tf.reduce_mean(is_correct)
        losses = []
        acc = []
        self.s.run(tf.global_variables_initializer())
        data = np.matrix(data)
        for i in range(steps):
            ind = np.random.choice(len(data), size=batch_size)
            train_x = data[ind]
            train_y = np.matrix(labels[ind]).T
            self.s.run(optimize, feed_dict={self.x: train_x, self.yy: train_y})
            loss = self.s.run(self.J, feed_dict={self.x: train_x, self.yy: train_y})
            a = self.s.run(accuracy, feed_dict={self.x: data, self.yy: np.matrix(labels).T})
            losses.append(loss)
            acc.append(a)
            print("Step:{}, Loss:{}, Accuracy:{}".format(i, loss, a))

        if graph:
            plt.plot(range(len(losses)), losses)
            plt.show()
