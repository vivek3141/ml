from ml.cnn import CNN2D
import pickle
import numpy as np

with open("label.txt", "rb") as f:
    label = pickle.load(f)
with open("data.txt", "rb") as f:
    image = pickle.load(f)

c = CNN2D(
    layers=2,
    inp=784,
    out=10,
    kernel_size=[5, 5],
    pool_size=[2, 2],
    filters=[32, 64],
    dimensions=[28, 28],
)

c.fit(data=np.array([image, image]), labels=np.array([label, label]), lr=0.001, epochs=1, save_path="./model")
c.test(data=np.array([image]), labels=np.array([label], dtype=np.int32))
predicted = c.predict(np.array([image]), transpose=True)
