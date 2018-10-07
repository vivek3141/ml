import numpy as np
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


def m(ar):
    print("hi")
    return False


def g(ar):
    print("bye")


if __name__ == '__main__':
    tf.app.run(m)
    print("hi")
    tf.app.close()
    tf.app.run(g)
