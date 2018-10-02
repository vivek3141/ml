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
        start = time.time()

        points, cluster_assignments = create_dataset(self.n)

        # Silly initialization:  Use the first two points as the starting
        # centroids.  In the real world, do this better.
        centroids = tf.Variable(tf.slice(points.initialized_value(), [0, 0], [self.k, 2]))

        # Replicate to N copies of each centroid and K copies of each
        # point, then subtract and compute the sum of squared distances.
        rep_centroids = tf.reshape(tf.tile(centroids, [self.n, 1]), [self.n, self.k, 2])
        rep_points = tf.reshape(tf.tile(points, [1, self.k]), [self.n, self.k, 2])
        sum_squares = tf.reduce_sum(tf.square(rep_points - rep_centroids),
                                    reduction_indices=2)

        # Use argmin to select the lowest-distance point
        best_centroids = tf.argmin(sum_squares, 1)
        did_assignments_change = tf.reduce_any(tf.not_equal(best_centroids,
                                                            cluster_assignments))

        means = self._bucket_mean(points, best_centroids, self.k)

        # Do not write to the assigned clusters variable until after
        # computing whether the assignments have changed - hence with_dependencies
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
        end = time.time()
        print(("Found in %.2f seconds" % (end - start)), iters, "iterations")
        print("Centroids:")
        print(centers)
        print("Cluster assignments:", assignments)
        return centers, assignments
