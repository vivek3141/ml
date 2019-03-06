#include <python3.6/Python.h>


static PyObject * optimize(PyObject *self, PyObject *args){
    int* theta;
    int learning_rate;
    int steps;
    int* init_theta;
    int dx;
    int num_theta;
    char* input;
    return Py_BuildValue("s", "Hello");
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


PyMODINIT_FUNC PyInit_gradient_descent(void)
{
    return PyModule_Create(&optimize);
}