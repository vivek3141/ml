from ml.cnn import CNN2D
from ml.data import mnist
from ml.data import get_predicted_value

c = CNN2D(
    layers=2,
    inp=784,
    out=10,
    kernel_size=[5, 5],
    pool_size=[2, 2],
    filters=[32, 64],
    dimensions=[28, 28],
)

c.fit(data=mnist.train.images, labels=mnist.train.labels, lr=0.001, epochs=200, save_path="./model")

# c.load("./model")

c.test(data=mnist.test.images[0:100], labels=mnist.test.labels[0:100])

predicted = c.predict(mnist.test.images[0], transpose=True)

print("Predicted Value: {}".format(get_predicted_value(predicted['probabilities'])))
print("Actual Value: {}".format(mnist.test.labels[0]))
