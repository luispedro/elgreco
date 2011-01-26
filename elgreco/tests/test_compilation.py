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
    assert value0 == data[0]
    v.value = 1+value0
    assert 1+value0 == data[0]
    assert 1+value0 == v.value
    gibbs.gibbs(g, 2)
