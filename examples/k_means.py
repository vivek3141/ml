from ml.k_means import KMeans
from ml.random.kmeans import create_dataset
from ml.graph.kmeans import plot

N = 1000
K = 4
data, y, p = create_dataset(N)
k = KMeans(K, N)
c, assign = k.fit(data, y, 50000)
plot(p, assign, c)
