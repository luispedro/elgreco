# Copyright (C) 2011, Luis Pedro Coelho <lpc@cmu.edu>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
#
# LICENSE: GPLv3

'''
===========
Compilation
===========

'''
from collections import defaultdict
import numpy as np
from elgreco.graph import Node

_module_header = '''
extern "C" {
    #include <Python.h>
    #include <numpy/ndarrayobject.h>
}
#include <limits>

namespace{

const char ErrorMsg[] =
    "El Greco internal error. Please report this as a bug.";


struct random_source {
    float uniform01() {
        return random();
    }
    float sample_gamma(float n) {
        return uniform01();
    }
};

PyObject* py_gibbs(PyObject* self, PyObject* args) {
    PyArrayObject* array;
    if (!PyArg_ParseTuple(args,"O", &array)) return NULL;
    if (!PyArray_Check(array) || PyArray_TYPE(array) != NPY_FLOAT || !PyArray_ISCARRAY(array)) {
        PyErr_SetString(PyExc_RuntimeError, ErrorMsg);
        return NULL;
    }
    float* data = static_cast<float*>(PyArray_DATA(array));
    random_source R;

'''

_module_footer = '''

    Py_INCREF(Py_None);
    return Py_None;
}

PyMethodDef methods[] = {
  {"gibbs",(PyCFunction)py_gibbs, METH_VARARGS, NULL},
  {NULL, NULL,0,NULL},
};

} // namespace
extern "C"
void init_elgreco()
  {
    import_array();
    (void)Py_InitModule("_elgreco", methods);
  }
'''


class Variables(object):
    def __init__(self):
        self.used = defaultdict(int)
        self.decls = []

    def allocate_float(self, basename, n=None):
        return self.allocate(basename, 'float', n)

    def allocate_int(self, basename, n=None):
        return self.allocate(basename, 'int', n)

    def allocate(self, basename, typename, n=None):
        postfix = typename[0]
        if n is not None:
            postfix += 'a'
        name = '%s_%s_%s' % (basename, self.used[basename], postfix)
        self.used[basename] += 1
        if n is None:
            self.decls.append('%s %s;' % (typename, name))
        else:
            self.decls.append('%s %s[%s];' % (typename, name, int(n)))
        return name

class ArrayRef(object):
    def __init__(self, array, idx, n=None):
        self.array = array
        if n is None:
            self.slice = int(idx)
        else:
            self.slice = slice(idx, idx+n)

    def get(self):
        return self.array[self.slice]

    def set(self, value):
        self.array[self.slice] = value

    def variable_name(self):
        if type(self.slice) == int:
            return '(data[%s])' % self.slice
        return '(data+%s)' % self.slice.start

    def delete(self, obj):
        raise NotImplementedError('elgreco.compilation.ArrayRef.__delete__')

class NodeRef(Node):
    def getval(self):
        return self._value.get()
    def setval(self, v):
        self._value.set(v)
    value = property(getval, setval)

def arrayify(graph):
    data = np.zeros(sum(v.model.size for v in graph.vertices), dtype=np.float32)
    next = 0
    for v in graph.vertices:
        size = v.model.size
        ref = ArrayRef(data, next, (size if size > 1 else None))
        if hasattr(v, 'value'):
            ref.set(v.value)
        v._value = ref
        v.__class__ = NodeRef
        next += size
    return data
