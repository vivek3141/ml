#include <math.h>

struct optimizer{
    int[] theta;
    
}
int[] optimize(int learning_rate, int steps, int[] init_theta, int dx){
    for(int i = 0; i < steps; i ++){
        partials = []
        for t in range(self.num_theta):
            theta_dx = [(theta[x] + dx) if t == x else theta[x] for x in range(self.num_theta)]
            partial = (self.func(*theta_dx) - self.func(*theta)) / dx
            partials.append(partial)
        for k in range(self.num_theta):
            theta[k] -= learning_rate * partials[k]
        if i % 50 == 0:
            print(f"Step: {i} Cost {self.func(*theta)}")
    return theta
}