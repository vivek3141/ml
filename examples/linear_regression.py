from ml.linear_regression import LinearRegression

# data
x = sorted([0.1, 2.3, 6.3, 9.2, 1.6, 2.3, 5.1, 7.2, 4.3, 1.5, 6.3, 8.9, 3.2, 3.5, 5.6])
y = list(map(lambda x: 0.3*x + 0.2, x))

l = LinearRegression()
l.fit(data=x, labels=y, graph=True)
