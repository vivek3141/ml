import matplotlib.pyplot as plt


def plot(data, assignments, centers):
    li = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for k, i in enumerate(data):
        plt.scatter(i[0], i[1], color=li[assignments[k] % len(li)])
    plt.scatter(centers[:][0], centers[:][1], color="black")
    plt.show()
