from ml.k_means import KMeans
from ml.random.kmeans import create_dataset

N = 1000
K = 13
data, y = create_dataset(N)

k = KMeans(K, N)
centers, assign = k.fit(
    points=data, cluster_assignments=y, epochs=10000, graph=True)
