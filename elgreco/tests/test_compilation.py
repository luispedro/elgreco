import numpy as np

from elgreco.systems import lda
from elgreco import gibbs
from elgreco.compilation import arrayify

def test_arrayify():
    K = 2
    documents = [
        (2,2,2,2,3,3,3,4,4,1),
        (2,3,2,3,4,2,3),
        (4,4,1,1,1,4,1,4),
    ]
    g = lda(documents, K)
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
