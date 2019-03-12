#include <python3.6/Python.h>

int call_func(PyObject* func, double* theta, int num_theta);

int * _optimize(PyObject* func, double learning_rate, int steps, double* init_theta,  
                double dx, int num_theta)
{
    int* theta = init_theta;
    for(int i = 0; i < steps; i ++)
    {
        int * partials = malloc(sizeof(int) * num_theta);
        for (int t=0; t < num_theta; t++)
        {
            int * theta_dx = malloc(sizeof(int) * num_theta);
            for( int x = 0; x < num_theta; x++)
            {
                if(t == x)
                {
                    theta_dx[x] = theta[x] + dx;
                }
                else
                {
                    theta_dx[x] = theta[x];
                }
            }
            partials[t] = call_func(func, theta, num_theta) - call_func(func, theta, num_theta) / dx;
        }
        for(int k = 0; k < num_theta; k++)
        {
            theta[k] -= learning_rate * partials[k];
        }
        if (i % 50 == 0)
        {
            printf("Step: {%d} Cost {%d}", i, call_func(func, theta, num_theta));
        }
    return theta;
    }
}

int call_func(PyObject* func, double* theta, int num_theta)
{
    PyObject* result = PyObject_CallObject(func, theta);


    /*const char* s = PyString_AsString(result);
    printf("%s", s);*/

    PyObject* repr = PyObject_Repr(result);
    PyObject* str = PyUnicode_AsEncodedString(repr, "utf-8", "~E~");
    const char *bytes = PyBytes_AS_STRING(str);
    

    printf("REPR: %s\n", bytes);

    Py_XDECREF(repr);
    Py_XDECREF(str);
}

static PyObject * optimize(PyObject* self, PyObject* args)
{
    double learning_rate;
    int steps;
    PyObject* init_theta;
    double dx;
    int num_theta;
    PyObject* func;

    printf("In the function!\n");
    if (!PyArg_ParseTuple(args, "OiiOii", &func, &learning_rate, &steps, &init_theta, &dx, &num_theta))
        return NULL;

    double* theta = malloc(sizeof(double) * num_theta);
    
    PyObject* test;
    for(int i = 0; i < num_theta; i++)
    {
        test = PyList_GetItem(init_theta, i);
        PyObject* repr = PyObject_Repr(test);
        PyObject* str = PyUnicode_AsEncodedString(repr, "utf-8", "~E~");
        const char *bytes = PyBytes_AS_STRING(str);
        theta[i] = atof(bytes);
        printf("%d", theta[i]);
    }


    int ret = call_func(func, theta, num_theta);

    
    return Py_BuildValue("i", ret);
}

static char optimize_docs[] =
    "usage: optimize()\n";


static PyMethodDef module_methods[] = 
{
    {"optimize", (PyCFunction) optimize, 
     METH_VARARGS, optimize_docs},
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


PyMODINIT_FUNC PyInit_gradient_descent(void)
{
    return PyModule_Create(&gradient_descent);
}