import elgreco.models
from elgreco.graph import Graph, Node
import numpy as np

def test_gibbs():
    np.random.seed(10)

    g = Graph()
    p = Node(elgreco.models.ConstantModel(np.zeros(4)+.1))
    c = Node(elgreco.models.DirichletModel())
    v = Node(elgreco.models.CategoricalModel(4))
    g.vertices = (p,c,v)
    g.add_edge(p,c)
    g.add_edge(c,v)

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
