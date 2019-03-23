#include <Python.h>

double* _fit(PyObject* x, PyObject* y, double lr, int steps, PyObject* init_theta)
{

}

static PyObject * fit(PyObject* self, PyObject* args)
{
    PyObject* x;
    PyObject* y;
    PyObject* init_theta;
    double lr;
    int steps;

    if (!PyArg_ParseTuple(args, "OOdiO", &x, &y, &lr, &steps, &init_theta))
        return NULL;

    double* theta = malloc(sizeof(double) * 2);

    int i;
    for(i = 0; i < 2; i++)
    {
        theta[i] = PyFloat_AsDouble(PyList_GetItem(init_theta, (Py_ssize_t)i));
        printf("%f", theta[i]);
    }

    double* ret_theta = _fit(x, y, lr, steps, init_theta);

    PyObject* ret = PyTuple_New(2);

    for(i = 0; i < 2; i++)
    {
        PyTuple_SetItem(ret, i, PyFloat_FromDouble(ret_theta[i]));
    }

    return Py_BuildValue("O", ret);
}

static char fit_docs[] =
    "usage: fit(data, labels, lr, steps, init_theta)\n";


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


PyMODINIT_FUNC PyInit_linear_regression(void)
{
    return PyModule_Create(&linear_regression);
}