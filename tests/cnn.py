from ml.cnn import CNN2D
import numpy as np
from ml.data import mnist
import pickle
with open("data.txt", "wb") as f:
    pickle.dump(mnist.train.images[0],f)

c = CNN2D(
    layers=2,
    inp=4,
    out=4,
    kernel_size=[1, 1],
    pool_size=[1, 1],
    filters=[1, 1],
    dimensions=[2, 2],
)

c.fit(data=np.array([0.1, 0.1, 0.1, 0.1]), labels=np.array([0,0,0,0]), lr=0.001, epochs=1, save_path="./model")
c.test(data=np.array([0, 0, 0, 0]), labels=[0,0,0,0])
predicted = c.predict(np.array([0, 0, 0, 0]), transpose=True)
