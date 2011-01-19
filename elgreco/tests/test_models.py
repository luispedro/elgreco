from elgreco import models

def test_constant_model():
    for v in (3, 2.22, -1):
        c = models.ConstantModel(v)
        for i in xrange(10):
            assert c.sample1([], []) == v
        assert c.logP(v, []) == 0.
        assert c.logP(v+1, []) < -1000.
