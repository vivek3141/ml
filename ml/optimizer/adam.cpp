#include <iostream>
#include <Python/Python.h>

using namespace std;

const char newl = '\n';

double call_func(PyObject* func, double* theta, int num_theta);

double* _optimizer(PyObject* func, double lr, int steps, double* init_theta)
{
    for(int i = 0; i < steps; i++){
        if(i % 50 == 0){
            printf("");
        }
    }

    
}