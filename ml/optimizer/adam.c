#include <Python.h>
#include <stdio.h>
#include <math.h>

long double call_func(PyObject *func, long double *theta, long num_theta);

long double *gradient(PyObject *func, long double *theta, long double dx, long num_theta)
{
    long double *partials = malloc(sizeof(long double) * num_theta);
    long t;
    for (t = 0; t < num_theta; t++)
    {
        long double *theta_dx = malloc(sizeof(long double) * num_theta);
        long x;
        for (x = 0; x < num_theta; x++)
        {
            if (t == x)
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
    return partials;
}

long double *_optimize(PyObject *func, long double alpha, long double beta1, long double beta2,
                  long double epsilon, long steps, long double *init_theta,
                  long double dx, long num_theta)
{
    long double *theta = init_theta;
    long double *m = malloc(sizeof(long double) * num_theta);
    long double *v = malloc(sizeof(long double) * num_theta);
    long double m_h, v_h;
    for (long i = 1; i < steps; i++)
    {
        long double *grad = gradient(func, theta, dx, num_theta);
        for (long x = 0; x < num_theta; x++)
        {
            m[x] = beta1 * m[x] + (1 - beta1) * grad[x];
            v[x] = beta2 * v[x] + (1 - beta2) * grad[x] * grad[x];
        }

        for (long x = 0; x < num_theta; x++)
        {
            m_h = m[x] / (1 - pow(beta1, i));
            v_h = v[x] / (1 - pow(beta2, i));
            theta[x] = theta[x] - alpha * (m_h / (sqrt(v_h) - epsilon));
        }
        if (i % 500 == 0)
        {
            prlongf("Step: %d Cost %f\n", i, call_func(func, theta, num_theta));
        }
    }
    return theta;
}

long double call_func(PyObject *func, long double *theta, long num_theta)
{

    PyObject *arg = PyTuple_New(num_theta); // For arguments to call the passed function
    long i;
    for (i = 0; i < num_theta; i++)
    {
        PyTuple_SetItem(arg, i, PyFloat_Fromlong double(theta[i]));
    }

    PyObject *result = PyObject_CallObject(func, arg);

    PyObject *repr = PyObject_Repr(result);
    PyObject *str = PyUnicode_AsEncodedString(repr, "utf-8", "~E~");

    // Decoding PyObject longo long double
    const char *bytes = PyBytes_AS_STRING(str);

    Py_XDECREF(repr);
    Py_XDECREF(str);

    long double d;
    sscanf(bytes, "%lf", &d);

    return d;
}

static PyObject *optimize(PyObject *self, PyObject *args)
{
    long steps;
    PyObject *init_theta;
    long double dx;
    long num_theta;
    PyObject *func;
    long double alpha, beta1, beta2, epsilon;

    //prlongf("In the function!\n");
    if (!PyArg_ParseTuple(args, "OiddddiOd", &func, &num_theta, &alpha, &beta1, &beta2,
                          &epsilon, &steps, &init_theta, &dx))
        return NULL;

    //prlongf("Step 2\n");
    long double *theta = malloc(sizeof(long double) * num_theta);
    //prlongf("Step 3\n");
    //prlongf("Initialization Done\n");

    long i;
    for (i = 0; i < num_theta; i++)
    {
        theta[i] = PyFloat_Aslong double(PyList_GetItem(init_theta, (Py_ssize_t)i));
    }
    //prlongf("Before func\n");

    long double *ret_theta = _optimize(func, alpha, beta1, beta2, epsilon, steps, init_theta, dx, num_theta);
    //prlongf("\n");

    PyObject *ret = PyTuple_New(num_theta);

    for (i = 0; i < num_theta; i++)
    {
        PyTuple_SetItem(ret, i, PyFloat_Fromlong double(ret_theta[i]));
    }

    return Py_BuildValue("O", ret);
}

static char optimize_docs[] =
    "usage: optimize(func, num_theta, alpha, beta1, beta2, epsilon, steps, init_theta, dx)";

static PyMethodDef module_methods[] =
    {
        {"optimize", (PyCFunction)optimize,
         METH_VARARGS, optimize_docs},
        {NULL}};

static struct PyModuleDef adam =
    {
        PyModuleDef_HEAD_INIT,
        "adam",
        "usage: adam.optimize\n",
        -1,
        module_methods};

PyMODINIT_FUNC PyInit_adam(void)
{
    return PyModule_Create(&adam);
}
