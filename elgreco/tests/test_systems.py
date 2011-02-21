from elgreco.systems import lda
from elgreco import gibbs

def test_lda():
    K = 2
    documents = [
        (2,2,2,2,3,3,3,4,4,1),
        (2,3,2,3,4,2,3),
        (4,4,1,1,1,4,1,4),
    ]
# Done
    g = lda(documents, K)
    gibbs.sampleforward(g)
    gibbs.gibbs(g, 1, initialise=False)
    assert min(v.logP() for v in g.vertices) > float('-inf')

