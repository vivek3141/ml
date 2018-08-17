import tensorflow as tf
from ml.error import Error
import matplotlib.pyplot as plt


class NeuralNetwork:
    def __init__(self, layers, data, labels, activation, lr=0.001):
        try:
            self.activation = activation
        except KeyError:
            print("Invalid Activation Function")
        self.layers = layers
        self.input = len(data[0])
        self.output = len(labels[0])
        self.lr = lr
        self.data = data
        self.labels = labels
        self.W = []
        self.b = []
        self.x = tf.placeholder(tf.float32, shape=[None, self.input])
        self.yy = tf.placeholder(tf.float32, shape=[None, self.output])
        self.z = self.create_layer(0, self.input, self.layers, self.x, False)
        self.y = self.create_layer(1, self.layers, self.output, self.z, False)
        self.J = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.yy, logits=self.y))
        self.optim = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.min = self.optim.minimize(self.J)
        self.s = tf.Session()
        self.s.run(tf.global_variables_initializer())

    def fit(self, steps, graph=False, check=50):
        losses = []
        check_at = int(len(self.data)/2)
        x_c = self.data[check_at].reshape(1, self.input)
        y_c = self.labels[check_at].reshape(1, self.output)
        for i in range(steps):
            x_val = self.data[i].reshape(1, self.input)
            y_val = self.labels[i].reshape(1, self.output)
            if i % check == 0:
                losses.append(self.s.run(self.J, feed_dict={self.x: x_c, self.yy: y_c}))
            self.s.run(self.min, feed_dict={self.x: x_val, self.yy: y_val})
        if graph:
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

    @staticmethod
    def _check_length(arr1, arr2):
        if len(arr1) != len(arr2):
            raise Error("Length of the test arrays should be the same!")
        return False

    def predict(self, x):
        ret = []
        for i in range(len(x)):
            ret.append(self.s.run(self.y, feed_dict={self.x: x[i]}))
        return ret

    def test(self, test_x, test_y):
        self._check_length(test_x, test_y)
        num_c = 0
        for i in range(len(test_x)):
            r = self.s.run(self.y, feed_dict={self.x: test_x[i]})
            if r == test_y[i]:
                num_c += 1
        acc = num_c / len(test_x)
        print("Accuracy: {}".format(acc))
        return acc

    def save(self, file_name):
        saver = tf.train.Saver()
        saver.save(self.s, "./" + str(file_name))
        self.s.close()
