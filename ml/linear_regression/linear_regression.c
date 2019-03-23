#include <Python.h>

double* _fit(double* x, double* y, double lr, int steps, double* init_theta, int n)
{

}
double* _linear_r(double* x, double* label, double m, double b, int steps, double lr, int n)
{
    double cost;
    for(int i = 0; i < steps; i++){
        double y;
        double m_grad = 0;
        double b_grad = 0;

        cost = 0;

        for(int i = 0; i < n; i++){
            y = m * x[i] + b;
            cost += pow(label[i] - y, 2);
            m_grad += x[i] * (label[i] - y);
            b_grad += label[i] - y;
        }

        cost = cost/n;
        m_grad = m_grad * -(2/n);
        b_grad = b_grad * -(2/n);
        m = m - (lr * m_grad);
        b = b - (lr * b_grad);
    }
    double ret[3] = {m, b, cost};
    return ret;
}
    

double * matrix_sub(double* mat1, double* mat2)
{
    int len = sizeof(mat1)/sizeof(mat2);
    double * mat = malloc(sizeof(double) * len);
    for(int i = 0; i < len; i++){
        mat[i] = mat1[i] - mat2[i];
    }
    return mat;
}


double hypothesis(double* theta, double x)
{
    return theta[0] * x + theta[1];
}

static PyObject * fit(PyObject* self, PyObject* args)
{
    PyObject* x_;
    PyObject* y_;
    PyObject* init_theta;
    double lr;
    int steps;
    int m;

    if (!PyArg_ParseTuple(args, "OOdiOi", &x_, &y_, &lr, &steps, &init_theta, &m))
        return NULL;

    double* theta = malloc(sizeof(double) * 2);
    double* x = malloc(sizeof(double) * m);
    double* y = malloc(sizeof(double) * m);

    int i;
    for(i = 0; i < 2; i++)
    {
        theta[i] = PyFloat_AsDouble(PyList_GetItem(init_theta, (Py_ssize_t)i));
    }

    for(i = 0; i < m; i++)
    {
        x[i] = PyFloat_AsDouble(PyList_GetItem(x_, (Py_ssize_t)i));
        y[i] = PyFloat_AsDouble(PyList_GetItem(y_, (Py_ssize_t)i));
    }

    double* ret_theta = _fit(x, y, lr, steps, init_theta, m);

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