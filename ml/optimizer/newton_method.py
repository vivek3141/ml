import numdifftools as diff
import numpy as np
from inspect import signature

class NewtonMethodOptimizer:
    def __init__(self,function,x0,step=0.8,number_iterations=100):
        '''
        The initializer does the following: 
        function: The function to be optimized which takes as input (vector) and returns a (scalar)
        step: step size in Newton update step - can be modified by the user
        number_iterations: number of iterations for newton method to run - inputed by the user
        '''
        self.function=signature(function)
        self.x0=np.array(x0)
        self.number_iterations=number_iterations
        self.step=step
    
    def newton_method_(self):
        '''
        Multivariate Newton Method for a function and return the critical values 
        '''
        x_actual=self.x0
        #Approximate hessian and the gradient using the library 
        H=diff.Hessian(self.function)
        g=diff.Gradient(self.function)

        for i in range(self.number_iterations):
            x_increment=x_actual-self.step*np.dot(np.linalg.inv(H(x_actual)),g(x_actual))
            #make sure we haven't converged yet - the value of threshold of 0.000001 can be modified 
            if abs(max(x_increment-x_actual))<0.000001:
                break
            x_actual=x_increment

        self.critical_x=x_increment
        self.values_x=self.function(x_actual)
        return self.critical_x
    
    def get_values(self):
        #This function must be called after newton_method
        return (self.critical_x, self.values_x)