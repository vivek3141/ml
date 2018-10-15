from ml.cnn import CNN2D
from ml.data import mnist

c = CNN2D(
    layers=2,
    inp=784,
    out=10,
    kernel_size=[5, 5],
    pool_size=[2, 2],
    filters=[32, 64],
    dimensions=[28, 28],
)
c.fit(data=mnist.train.images,
      labels=mnist.train.labels,
      lr=0.001,
      epochs=200,
      save_path="./CNN_model",)

# c.load("./CNN_model")
pred = c.predict(mnist.train.images[0], transpose=True)
print(pred)
