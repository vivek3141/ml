from ml.k_means import KMeans
from ml.random.kmeans import create_dataset

N = 4
K = 4
data, y = create_dataset(N)
k = KMeans(K, N)
centers, assign = k.fit(points=data, cluster_assignments=y, epochs=1, graph=False)
