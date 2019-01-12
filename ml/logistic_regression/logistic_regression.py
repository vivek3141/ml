import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression:
    def __init__(self, features, seed=5):
        """
        Does logistic regression
        :param features: number of features
        :param seed: seed for randomly initializing the values
        """
        self.input = features
        tf.set_random_seed(seed=seed)
        self.W = tf.Variable(tf.zeros(shape=[self.input, 1]))
        self.b = tf.Variable(tf.zeros(shape=[1]))
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.input])
        self.yy = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.y = tf.matmul(self.x, self.W) + self.b
        self.predicted_no_round = tf.sigmoid(self.y)
        self.predicted = tf.round(tf.sigmoid(self.y))
        self.s = tf.Session()
        self.J = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.y, labels=self.yy))

    def fit(self, data, labels, lr=0.001, steps=1000, graph=False, batch_size=25, to_print=True):
        """
        Fit the model from the given data
        :param data: Input data matrix
        :param labels: Input labels
        :param lr: Learning Rate
        :param steps: Number of steps
        :param graph: True if to graph the function
        :param batch_size: Batch size
        :param to_print: True if to print the loss after a certain amount of steps
        :return: None
        """
        minimize = tf.train.AdamOptimizer(learning_rate=lr)
        optimize = minimize.minimize(self.J)
        losses = []
        self.s.run(tf.global_variables_initializer())
        data = np.matrix(data)
        for i in range(steps):
            ind = np.random.choice(len(data), size=batch_size)
            train_x = data[ind]
            train_y = np.matrix(labels[ind]).T
            self.s.run(optimize, feed_dict={self.x: train_x, self.yy: train_y})
            loss = self.s.run(self.J, feed_dict={self.x: train_x, self.yy: train_y})
            losses.append(loss)
            if steps % 100 == 0 and to_print:
                print("Step:{}, Loss:{}".format(i, loss))

        if graph:
            plt.plot(range(len(losses)), losses)
            plt.show()

    def test(self, data, labels):
        """
        Test the model for accuracy
        :param data: Input data
        :param labels: Input labels
        :return: Accuracy (float)
        """
        correct = 0
        for i in range(len(data)):
            if labels[i] == self.predict(data[i]):
                correct += 1
        print("Accuracy: {}%".format(correct / len(data) * 100))
        return correct / len(data) * 100

    def hypothesis(self, data):
        """
        Run h(x) for some x
        :param data: Input data
        :return: Value for h(x)
        """
        return self.s.run(self.y, feed_dict={self.x: data})

    def predict(self, data, to_round=True):
        """
        Predict the value based on input data
        :param data: Input data
        :param to_round: Set true to round the value to 1 or 0
        :return: int or float for predicted value
        """
        if to_round:
            return self.s.run(self.predicted, feed_dict={self.x: np.matrix(data)})[0][0]
        return self.s.run(self.predicted_no_round, feed_dict={self.x: np.matrix(data)})[0][0]

    def get_values(self):
        """
        Get weights and biases
        :return: Weight vector and bias vector
        """
        return np.array(self.s.run(self.W)), self.s.run(self.b)[0]