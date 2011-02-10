# Copyright (C) 2011, Luis Pedro Coelho <lpc@cmu.edu>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
#
# LICENSE: GPLv3

'''
=====
Graph
=====

- Node
- Graph
'''

class Node(object):
    '''
    Node

    Attributes
    ----------
    children : list of Node
    parents : list of Node
    name : unicode
    model : elgreco.models
    value : anything, optional
    fixed : boolean
    '''
    def __init__(self, model, name=u'unnamed'):
        self.children = []
        self.parents = []
        self.name = name
        self.model = model
        self.fixed = False

    def logP(self):
        '''
        logp = n.logP()

        Return the log probability for the current value and the value of the
        parents.

        Returns
        -------
        logp : float
        '''
        return self.model.logP(self.value, self.parents)

    def sample1(self):
        '''
        value = n.sample1()

        Sample the node value given the current value of the parents and the
        children.

        '''
        if not self.fixed:
            self.value = self.model.sample1(self, self.parents, self.children)
        return self.value

    def sampleforward(self):
        '''
        value = n.sampleforward()

        Sample the value given the value of its parents (independent of the
        children!).
        '''
        if not self.fixed:
            self.value = self.model.sampleforward(self, self.parents)
        return self.value

    def fix(self, value):
        '''
        n.fix(v)

        Fix the value of n to v
        '''
        self.value = value
        self.fixed = True

    def __unicode__(self):
        return 'N[%s]' % self.name

class Graph(object):
    '''
    Graph

    Attributes
    ----------
    vertices : set of nodes
    '''
    def __init__(self):
        '''
        Constructs an empty graph
        '''
        self.vertices = set()

    def add_edge(self, n0, n1):
        '''
        g.add_edge(n0, n1)

        Add edge ``n0 -> n1``

        Parameters
        ----------
        n0 : Node
        n1 : Node
        '''
        n0.children.append(n1)
        n1.parents.append(n0)
        self.vertices.add(n0)
        self.vertices.add(n1)

    def add_edges(self, n0s, n1s):
        '''
        g.add_edges([a0,a1,a2], [b0,b1])

        Creates edges between every element of the first set and every element
        of the second.

        In the above example it creates the edges a0->b0, a1->b0, a2->b0,
        a0->b1, a1->b1, and a2->b1.

        Parameters
        ----------
        n0s : iterable of Node
        n1s : iterable of Node

        See Also
        --------
        add_edge : function
            ``add_edges`` is just a convenience function around ``add_edge()``
        '''
        for n0 in n0s:
            for n1 in n1s:
                self.add_edge(n0, n1)

    def roots(self):
        '''
        roots = g.roots()

        A root is a node with no parents. This function returns a list of them.

        Returns
        -------
        roots : list of Node
        '''
        return [n for n in self.vertices if not n.parents]

