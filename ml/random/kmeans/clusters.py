import tensorflow as tf


def create_dataset(n):
    points = tf.Variable(tf.random_uniform([n, 2]))
    cluster_assignments = tf.Variable(tf.zeros([n], dtype=tf.int64))
    return points, cluster_assignments
