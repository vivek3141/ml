from ml.optimizer import GradientDescentOptimizer
from ml.optimizer import NewtonMethodOptimizer


g = GradientDescentOptimizer(lambda _: 0)
theta = g.optimize(steps=1)

f=NewtonMethodOptimizer(lambda _: 0, 0)
theta_bis=f.newton_method_()