from inspect import signature
import gradient_descent


class GradientDescentOptimizer:
    def __init__(self, func, num_theta=None):
        self.func = func
        self.num_theta = len(signature(func).parameters) if num_theta is None else num_theta

    def optimize(self, learning_rate=0.01, steps=10000, init_theta=None, dx=0.0001):
        theta = [0 for _ in range(self.num_theta)
                 ] if init_theta is None else init_theta
        return gradient_descent.optimize(self.func, learning_rate, int(steps), theta, dx, int(self.num_theta))

    def optimize_python(self, learning_rate=0.01, steps=10000, init_theta=None, dx=0.0001):
        theta = [0 for _ in range(self.num_theta)
                 ] if init_theta is None else init_theta
        for i in range(steps):
            partials = []
            for t in range(self.num_theta):
                theta_dx = [(theta[x] + dx) if t == x else theta[x]
                            for x in range(self.num_theta)]
                partial = (self.func(*theta_dx) - self.func(*theta)) / dx
                partials.append(partial)
            for k in range(self.num_theta):
                theta[k] -= learning_rate * partials[k]
            if i % 50 == 0:
                print(f"Step: {i} Cost {self.func(*theta)}")
        return theta
