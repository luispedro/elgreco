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
#include <cstring>
#include <cmath>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

namespace{

const char ErrorMsg[] =
    "El Greco internal error. Please report this as a bug.";


struct random_source {
    random_source(unsigned s)
        :r(gsl_rng_alloc(gsl_rng_mt19937))
    {
        gsl_rng_set(r, s);
    }
    ~random_source() {
        gsl_rng_free(r);
    }


    float uniform01() {
        return gsl_ran_flat(r, 0., 1.);
    }
    float gamma(float a, float b) {
        return gsl_ran_gamma(r, a, b);
    }
    private:
       gsl_rng * r;
};
'''

_function_start = '''
PyObject* py_gibbs(PyObject* self, PyObject* args) {
    PyArrayObject* array;
    unsigned seed;
    if (!PyArg_ParseTuple(args,"Oi", &array, &seed)) return NULL;
    if (!PyArray_Check(array) || PyArray_TYPE(array) != NPY_FLOAT || !PyArray_ISCARRAY(array)) {
        PyErr_SetString(PyExc_RuntimeError, ErrorMsg);
        return NULL;
    }
    float* data = static_cast<float*>(PyArray_DATA(array));
    random_source R(seed);

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
    gsl_rng_env_setup();
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

def output_code(output, g):
    '''
    output_code(output, g)

    Output code to file-like `output`

    Parameters
    ----------
    output : file-like
    g : Node.Graph
        Graph that has been arrayify()ed
    '''
    print >>output, _module_header
    for v in g.vertices:
        v.model = v.model.compiled()
    headers = set()
    for v in g.vertices:
        if v.fixed: continue
        headers.add(tuple(v.model.headers()))

    for head in headers:
        for h in head:
            print >>output, h
    print >>output, _function_start
    for v in g.vertices:
        if v.fixed: continue
        for code in v.model.sample1(v, v.parents, v.children):
            print >>output, code
    print >>output, _module_footer

