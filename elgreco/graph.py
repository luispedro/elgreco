# Copyright (C) 2011, Luis Pedro Coelho <lpc@cmu.edu>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
# 
# LICENSE: GPLv3

class Node(object):
    def __init__(self, model):
        self.children = []
        self.parents = []
        self.name = u'unnamed'
        self.model = model

    def logP(self):
        return self.model.logP(self.value, self.parents)
        
    def __unicode__(self):
        return 'N[%s]' % self.name

class Graph(object):
    def __init__(self):
        self.vertices = []

    def add_edge(self, n0, n1):
        n0.children.append(n1)
        n1.parents.append(n0)
