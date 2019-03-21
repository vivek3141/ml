#include <Python.h>

double call_func(PyObject* func, double* theta, int num_theta);

double * _optimize(PyObject* func, double learning_rate, int steps, double* init_theta,  
                double dx, int num_theta)
{
    double* theta = init_theta;
    int i; // Initialize variable here to deal with outdated compilers
    for(i = 0; i < steps; i++)
    {
        double * partials = malloc(sizeof(double) * num_theta);
        int t;
        for (t = 0; t < num_theta; t++)
        {
            double * theta_dx = malloc(sizeof(double) * num_theta);
            int x;
            for(x = 0; x < num_theta; x++)
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
            partials[t] = (call_func(func, theta_dx, num_theta) - call_func(func, theta, num_theta)) / dx;
        }
        int k;
        for(k = 0; k < num_theta; k++)
        {
            theta[k] -= learning_rate * partials[k];
        }
        if (i % 500 == 0)
        {
            printf("Step: %d Cost %f\n", i, call_func(func, theta, num_theta));
        }
    }
    return theta;
}

double call_func(PyObject* func, double* theta, int num_theta)
{
    
    PyObject* arg = PyTuple_New(num_theta); // For arguments to call the passed function
    int i;
    for(i = 0; i < num_theta; i++)
    {
        PyTuple_SetItem(arg, i, PyFloat_FromDouble(theta[i]));
    }

    PyObject* result = PyObject_CallObject(func, arg);

    PyObject* repr = PyObject_Repr(result);
    PyObject* str = PyUnicode_AsEncodedString(repr, "utf-8", "~E~");

    // Decoding PyObject into double
    const char *bytes = PyBytes_AS_STRING(str);

    Py_XDECREF(repr);
    Py_XDECREF(str);

    double d;
    sscanf(bytes, "%lf", &d);

    return d;

}

static PyObject * optimize(PyObject* self, PyObject* args)
{
    double learning_rate;
    int steps;
    PyObject* init_theta;
    double dx;
    int num_theta;
    PyObject* func;

    //printf("In the function!\n");
    if (!PyArg_ParseTuple(args, "OdiOdi", &func, &learning_rate, &steps, &init_theta, &dx, &num_theta))
        return NULL;
    
    //printf("Step 2\n");
    double* theta = malloc(sizeof(double) * num_theta);
    //printf("Step 3\n");
    //printf("Initialization Done\n");

    int i;
    for(i = 0; i < num_theta; i++)
    {
        theta[i] = PyFloat_AsDouble(PyList_GetItem(init_theta, (Py_ssize_t)i));
    }
    //printf("Before func\n");

    double* ret_theta = _optimize(func, learning_rate, steps, theta, dx, num_theta);
    //printf("\n");

    PyObject* ret = PyTuple_New(num_theta);

    for(i = 0; i < num_theta; i++)
    {
        PyTuple_SetItem(ret, i, PyFloat_FromDouble(ret_theta[i]));
    }

    return Py_BuildValue("O", ret);
}

static char optimize_docs[] =
    "usage: optimize(func, learning_rate, steps, init_theta, dx, num_theta)\n";


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