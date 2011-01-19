import numpy as np
from elgreco.graph import Graph, Node
import elgreco.models
def test_add_edge():
    m = None
    g = Graph()
    p = Node(m)
    n = Node(m)
    c0 = Node(m)
    c1 = Node(m)
    g.add_edge(p, n)
    g.add_edge(n, c0)
    g.add_edge(n, c1)

    assert n in p.children
    assert p in n.parents

    assert c0 in n.children
    assert n in c0.parents

    assert c1 in n.children
    assert n in c1.parents

def test_model_graph():
    p = Node(elgreco.models.ConstantModel((.3, .7)))
    p.sample1()
    assert p.value == (.3, .7)

