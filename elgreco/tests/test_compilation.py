import numpy as np

from elgreco.systems import lda
from elgreco import gibbs
from elgreco.compilation import arrayify

def lda_simple():
    K = 2
    documents = [
        (2,2,2,2,3,3,3,4,4,1),
        (2,3,2,3,4,2,3),
        (4,4,1,1,1,4,1,4),
    ]
    g = lda(documents, K)
    return g

def test_arrayify():
    g = lda_simple()
    gibbs.sampleforward(g)
    for v in g.vertices:
        break

    value0 = v.value
    data = arrayify(g)
    # We need to convert to the same type, otherwise we get rounding errors:
    assert np.all(value0.astype(data.dtype) == data[:len(value0)])
    v.value = 1+value0
    assert np.all((1.+value0).astype(data.dtype) == data[:len(value0)])
    assert np.all((1+value0).astype(v.value.dtype) == v.value)
    gibbs.gibbs(g, 2)
    assert min(v.logP() for v in g.vertices) > float('-inf')

def test_data_usage():
    g = lda_simple()
    data = arrayify(g)
    counts = np.zeros_like(data)
    for v in g.vertices:
        counts[v._value.slice] += 1

    assert np.all(counts == 1.)

