#include <python3.6/Python.h>

static PyObject* uniquetest(PyObject* self)
{
    return Py_BuildValue("s", "ur mom gay");
}

static char uniquetest_docs[] =
    "usage: uniquetest(lstSortableItems, comboSize)\n";


static PyMethodDef module_methods[] = {
    {"uniquetest", (PyCFunction) uniquetest, 
     METH_NOARGS, uniquetest_docs},
    {NULL}
};


static struct PyModuleDef test =
{
    PyModuleDef_HEAD_INIT,
    "test", 
    "usage: test.uniquetest(lstSortableItems, comboSize)\n", /* module documentation, may be NULL */
    -1,   /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    module_methods
};

PyMODINIT_FUNC PyInit_test(void)
{
    return PyModule_Create(&test);
}