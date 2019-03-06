#include <python3.6/Python.h>

int * _optimize(int (func)(int*), int learning_rate, int steps, int* init_theta, int dx, int num_theta){
    int* theta = init_theta;
    for(int i = 0; i < steps; i ++){
        int * partials = malloc(sizeof(int) * num_theta);
        for (int t=0; t < num_theta; t++){
            int * theta_dx = malloc(sizeof(int) * num_theta);
            for( int x = 0; x < num_theta; x++){
                if(t == x){
                    theta_dx[x] = theta[x] + dx;
                }
                else{
                    theta_dx[x] = theta[x];
                }
            }
            partials[t] = func(theta) - func(theta) / dx;
        }
        for(int k = 0; k < num_theta; k++){
            theta[k] -= learning_rate * partials[k];
        }
        if (i % 50 == 0){
            printf("Step: {%d} Cost {%d}", i, func(theta));
        }
    return theta;
    }
}

static PyObject * optimize(PyObject *self, PyObject *args){
    int* theta;
    int learning_rate;
    int steps;
    int* init_theta;
    int dx;
    int num_theta;
    char* input;
    if(!PyArg_ParseTuple(args, "s", &input))
        return NULL;
    return Py_BuildValue("s", print(input));
}

static char optimize_docs[] =
    "usage: optimize()\n";


static PyMethodDef module_methods[] = {
    {"optimize", (PyCFunction) optimize, 
     METH_NOARGS, optimize_docs},
    {NULL}
};


static struct PyModuleDef gradient_descent =
{
    PyModuleDef_HEAD_INIT,
    "gradient_descent", 
    "usage: gradient_descent.optimize\n", 
    -1,   
    module_methods
};


PyMODINIT_FUNC PyInit_optimize(void)
{
    return PyModule_Create(&optimize);
}