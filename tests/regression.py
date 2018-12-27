from ml.regression import Regression

x1 = [0]
y1 = [0]

r = Regression("a,x:a*x")
t = r.fit(x1, y1, to_print=50, steps=1)
