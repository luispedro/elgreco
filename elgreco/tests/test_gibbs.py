import elgreco.models
from elgreco.graph import Graph, Node
from elgreco import gibbs
import numpy as np


def pcv_graph():
    g = Graph()
    p = Node(elgreco.models.ConstantModel(np.zeros(4)+.1))
    c = Node(elgreco.models.DirichletModel())
    v = Node(elgreco.models.CategoricalModel(4))
    g.vertices = (p,c,v)
    g.add_edge(p,c)
    g.add_edge(c,v)
    return g,p,c,v


def test_sampleforward():
    g,p,c,v = pcv_graph()
    gibbs.sampleforward(g, 0)
    values = []
    for n in (p,c,v):
        assert hasattr(n, 'value')
        values.append(n.value)
    gibbs.sampleforward(g, 0)
    for n,pv in zip((p,c,v), values):
        assert np.all(n.value == pv)


def test_gibbs_by_hand():
    np.random.seed(10)
    g,p,c,v = pcv_graph()

    p.sample1()
    v.value = 3
    c.sample1()

    samples = []
    for iter in xrange(1000):
        for n in g.vertices:
            n.sample1()
        samples.append(v.value)
    samples = np.array(samples)

    counts = [(samples == i).sum() for i in xrange(4)]
    counts = np.array(counts)
    assert counts.min() > 220
    assert counts.max() < 280

def test_gibbs():
    np.random.seed(10)
    g,p,c,v = pcv_graph()
    samples = gibbs.gibbs(g, 1000, sample_function=(lambda g: v.value))
    samples = np.array(samples)

    counts = [(samples == i).sum() for i in xrange(4)]
    counts = np.array(counts)
    assert counts.min() > 220
    assert counts.max() < 280


