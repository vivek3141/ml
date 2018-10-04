import tensorflow as tf


def create_dataset(n):
    points = tf.Variable(tf.random_uniform([n, 2]))
    cluster_assignments = tf.Variable(tf.zeros([n], dtype=tf.int64))
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    p = sess.run(points)
    return points, cluster_assignments, p
