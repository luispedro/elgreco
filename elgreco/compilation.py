# Copyright (C) 2011, Luis Pedro Coelho <lpc@cmu.edu>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
#
# LICENSE: GPLv3

'''
===========
Compilation
===========

'''
from elgreco.graph import Node

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

    def delete(self, obj):
        raise NotImplementedError('elgreco.compilation.ArrayRef.__delete__')

class NodeRef(Node):
    def getval(self):
        return self._value.get()
    def setval(self, v):
        self._value.set(v)
    value = property(getval, setval)

def arrayify(graph):
    data = np.zeros(sum(v.model.size for v in graph.vertices))
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
