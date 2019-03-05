#include <math.h>
#include <stdio.h>

int * optimize(void (*func)(int), int learning_rate, int steps, int* theta, int dx, int num_theta){
    for(int i = 0; i < steps; i ++){
        int * partials = malloc(sizeof(int) * num_theta);
        for (int t=0; t < num_theta; t++){
            int * theta_dx = malloc(sizeof(int) * num_theta);
            for( int x = 0; x < num_theta; x++){
                if(t==x){
                    theta_dx[x] = theta[x] + dx;
                }
                else{
                    theta[x];
                }
            }
            partials[t] = (*func)(theta_dx) - (*func)(theta) / dx;
        }
        for k in range(self.num_theta){
            theta[k] -= learning_rate * partials[k]
        }
        if i % 50 == 0{
            printf(f"Step: {i} Cost {self.func(*theta)}")
        }
    return theta
}