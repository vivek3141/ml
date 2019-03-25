#include <Python.h>

double* _fit(double* x, double* label, double* init_theta, int steps, double lr, int n)
{
    double m = init_theta[0];
    double b = init_theta[1];
    double cost;
    int i;

    for(i = 0; i < steps; i++){
        double y;
        double m_grad = 0;
        double b_grad = 0;

        cost = 0;

        int k;
        for(k = 0; i < n; i++){
            y = m * x[i] + b;
            cost += pow(label[i] - y, 2);
            m_grad += x[i] * (label[i] - y);
            b_grad += label[i] - y;
        }

        cost = cost / n;
        
        m_grad = m_grad * (-2/(double)n);
        b_grad = b_grad * (-2/(double)n);
        

        m = m - (lr * m_grad);
        b = b - (lr * b_grad);

        if(i % 300 == 0){
            printf("Step: %d Cost %f\n", i, cost);
        }
    }
    double* ret = malloc(sizeof(double) * 2);
    ret[0] = m;
    ret[1] = b;

    return ret;
}
    

double * matrix_sub(double* mat1, double* mat2)
{
    int len = sizeof(mat1)/sizeof(mat2);
    double * mat = malloc(sizeof(double) * len);
    int i;
    for(i = 0; i < len; i++){
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

    double* ret_theta = _fit(x, y, theta, steps, lr, m);
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


static struct PyModuleDef lin_reg =
{
    PyModuleDef_HEAD_INIT,
    "lin_reg", 
    "usage: lin_reg.fit\n", 
    -1,   
    module_methods
};


PyMODINIT_FUNC PyInit_lin_reg(void)
{
    return PyModule_Create(&lin_reg);
}