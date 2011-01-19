from elgreco import models
import numpy as np

def test_constant_model():
    for v in (3, 2.22, -1):
        c = models.ConstantModel(v)
        for i in xrange(10):
            assert c.sample1([], []) == v
        assert c.logP(v, []) == 0.
        assert c.logP(v+1, []) < -1000.

def test_dirichlet_model():
    np.random.seed(2)
    d = models.DirichletModel()
    alphas = np.zeros(3)+.1
    children = [np.ones(3), np.ones(3)+3]
    samples = np.sum([d.sample1([alphas], children) for i in xrange(1000)], axis=0)
    assert samples.ptp() < 30

