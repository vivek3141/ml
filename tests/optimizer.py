from ml.optimizer import GradientDescentOptimizer

g = GradientDescentOptimizer(lambda _: 0)
theta = g.optimize(steps=1)
