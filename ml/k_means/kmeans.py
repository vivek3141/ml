import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
from ml.random.kmeans import create_dataset


class KMeans:
    def __init__(self, k):
        self.k = k
        self.sess = None
        self.n = None

    @staticmethod
    def _bucket_mean(data, bucket_ids, num_buckets):
        total = tf.unsorted_segment_sum(data, bucket_ids, num_buckets)
        count = tf.unsorted_segment_sum(tf.ones_like(data), bucket_ids, num_buckets)
        return total / count

    def train(self, data, epochs):
        self.n = len(data)
        points, cluster_assignments = create_dataset(self.n)
        centroids = tf.Variable(tf.slice(points.initialized_value(), [0, 0], [self.k, 2]))
        rep_centroids = tf.reshape(tf.tile(centroids, [self.n, 1]), [self.n, self.k, 2])
        rep_points = tf.reshape(tf.tile(points, [1, self.k]), [self.n, self.k, 2])
        sum_squares = tf.reduce_sum(tf.square(rep_points - rep_centroids),
                                    reduction_indices=2)
        best_centroids = tf.argmin(sum_squares, 1)
        did_assignments_change = tf.reduce_any(tf.not_equal(best_centroids,
                                                            cluster_assignments))
        means = self._bucket_mean(points, best_centroids, self.k)
        with tf.control_dependencies([did_assignments_change]):
            do_updates = tf.group(
                centroids.assign(means),
                cluster_assignments.assign(best_centroids))
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())
        changed = True
        iters = 0
        while changed and iters < epochs:
            iters += 1
            [changed, _] = self.sess.run([did_assignments_change, do_updates])
        [centers, assignments] = self.sess.run([centroids, cluster_assignments])
        return centers, assignments

    def save(self, file_name):
        saver = tf.train.Saver()
        saver.save(self.sess, "./" + str(file_name))
        self.sess.close()

    def load(self, file_name):
        load = tf.train.Saver()
        self.sess = tf.Session()
        load.restore(self.sess, file_name)
