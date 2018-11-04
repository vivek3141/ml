from ml.data import mnist
import pickle
with open("label.txt","wb") as f:
    pickle.dump(mnist.train.labels[0], f)