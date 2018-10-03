from ml.k_means import KMeans
from ml.random.kmeans import create_dataset
from ml.graph.kmeans import plot

data, y = create_dataset(100)
k = KMeans(4)
c, assign = k.fit(data, 100)
plot(data, assign, c)
