import tensorflow as tf
from ml.graph.kmeans import plot


class KMeans:
    def __init__(self, k, n):
        self.k = k
        self.sess = None
        self.n = n

    @staticmethod
    def _bucket_mean(data, bucket_ids, num_buckets):
        total = tf.unsorted_segment_sum(data, bucket_ids, num_buckets)
        count = tf.unsorted_segment_sum(tf.ones_like(data), bucket_ids, num_buckets)
        return total / count

    def fit(self, points, cluster_assignments, epochs, graph=False):
        """
        Takes input to fit model
        :param points: Data points matrix
        :param cluster_assignments: Initial Cluster assignments
        :param epochs: Number of steps
        :param graph: True if to graph the data points
        :return: Centers and cluster assignments
        """
        centroids = tf.Variable(tf.slice(points.initialized_value(), [0, 0], [self.k, 2]))
        rep_centroids = tf.reshape(tf.tile(centroids, [self.n, 1]), [self.n, self.k, 2])
        rep_points = tf.reshape(tf.tile(points, [1, self.k]), [self.n, self.k, 2])
        sum_squares = tf.reduce_sum(tf.square(rep_points - rep_centroids),
                                    reduction_indices=2)
        best_centroids = tf.argmin(sum_squares, 1)
        did_assignments_change = tf.reduce_any(tf.not_equal(best_centroids,
                                                            cluster_assignments))
        means = KMeans._bucket_mean(points, best_centroids, self.k)
        with tf.control_dependencies([did_assignments_change]):
            do_updates = tf.group(
                centroids.assign(means),
                cluster_assignments.assign(best_centroids))
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        changed = True
        iters = 0
        while changed and iters < epochs:
            iters += 1
            [changed, _] = self.sess.run([did_assignments_change, do_updates])
        [centers, assignments] = self.sess.run([centroids, cluster_assignments])
        data = self.sess.run(points)
        if graph:
            plot(data, assignments, centers)
        return centers, assignments

    def save(self, file_name):
        """
        Save the model
        :param file_name: File to save the model to
        :return: None
        """
        saver = tf.train.Saver()
        saver.save(self.sess, "./" + str(file_name))
        self.sess.close()

    def load(self, file_name):
        """
        Load the model
        :param file_name: File to load the model from
        :return: None
        """
        load = tf.train.Saver()
        self.sess = tf.Session()
        load.restore(self.sess, file_name)
