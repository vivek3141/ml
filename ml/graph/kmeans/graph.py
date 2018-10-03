import matplotlib.pyplot as plt


def plot(data, assignments, centers):
    li = ["red", "blue", "yellow", "green"]
    for k, i in enumerate(data):
        plt.scatter(i[0], i[1], color=li[assignments[k]])
    plt.scatter(centers[:][0], centers[:][1], color="black")
    plt.show()

