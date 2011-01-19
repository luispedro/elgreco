import numpy as np
from elgreco.graph import Graph, Node

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
