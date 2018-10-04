import matplotlib.pyplot as plt
import tensorflow as tf


def plot(data, assignments, centers):
    li = ["red", "blue", "yellow", "green"]
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    data = sess.run(data)
    for k, i in enumerate(data):
        plt.scatter(i[0], i[1], color=li[assignments[k]])
    plt.scatter(centers[:][0], centers[:][1], color="black")
    plt.show()

