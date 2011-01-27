from elgreco.graph import Graph, Node
from elgreco import models
import numpy as np

class SimpleNode(object):
    def __init__(self, value):
        self.value = value
        self.model = models.MultinomialModel(len(value))

def test_constant_model():
    for v in (3, 2.22, -1):
        c = models.ConstantModel(v)
        for i in xrange(10):
            assert c.sample1(None, [], []) == v
        assert c.logP(v, []) == 0.
        assert c.logP(v+1, []) < -1000.

def test_dirichlet_model():
    np.random.seed(2)
    d = models.DirichletModel(3)
    alphas = np.zeros(3)+.1
    children = map(SimpleNode, [np.ones(3), np.ones(3)+3])
    samples = np.sum([d.sample1(None, [SimpleNode(alphas)], children) for i in xrange(1000)], axis=0)
    assert samples.ptp() < 30


def test_binomial():
    p = Node(models.ConstantModel((.3, .7)))
    c = Node(models.BinomialModel())
    p.sample1()
    c.parents = [p]
    assert np.abs(sum(c.sample1() for i in xrange(1000))-700) < 100

