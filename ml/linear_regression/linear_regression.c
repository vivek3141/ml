#include <Python.h>

double* _fit()
{

}

static PyObject * fit(PyObject* self, PyObject* args)
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
    double* theta = malloc(sizeof(double) * 2);
    //printf("Step 3\n");
    //printf("Initialization Done\n");

    int i;
    for(i = 0; i < num_theta; i++)
    {
        theta[i] = PyFloat_AsDouble(PyList_GetItem(init_theta, (Py_ssize_t)i));
    }
    //printf("Before func\n");

    double* ret_theta = _fit(func, learning_rate, steps, theta, dx, num_theta);
    //printf("\n");

    PyObject* ret = PyTuple_New(2);

    for(i = 0; i < 2; i++)
    {
        PyTuple_SetItem(ret, i, PyFloat_FromDouble(ret_theta[i]));
    }

    return Py_BuildValue("O", ret);
}

static char fit_docs[] =
    "usage: fit(func, learning_rate, steps, init_theta, dx, num_theta)\n";


static PyMethodDef module_methods[] = 
{
    {"fit", (PyCFunction) fit, 
     METH_VARARGS, fit_docs},
    {NULL}
};


static struct PyModuleDef linear_regression =
{
    PyModuleDef_HEAD_INIT,
    "linear_regression", 
    "usage: linear_regression.fit\n", 
    -1,   
    module_methods
};


PyMODINIT_FUNC PyInit_gradient_descent(void)
{
    return PyModule_Create(&linear_regression);
}