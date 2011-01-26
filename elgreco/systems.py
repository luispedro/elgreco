# Copyright (C) 2011, Luis Pedro Coelho <lpc@cmu.edu>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
# 
# LICENSE: GPLv3

from .graph import Graph, Node
from . import models
import numpy as np

def lda(documents, K, alpha=.1):
    '''
    graph = lda(documents, K, alpha=.1)

    Build an LDA model for documents

    Parameters
    ----------
    documents : sequence of sequences of integers
        This is a collection of documents where each document is a sequence of
        words. Each word is an integer.
    K : integer
        Nr of topics
    alpha : float, optional
        Value for alpha hyperparameter

    Returns
    -------
    graph : elgreco.graph.Graph
        The LDA graphical model

    '''
    N = len(documents)
    Nwords = 1 + max(max(doc) for doc in documents)

    graph = Graph()
    alpha = Node(models.ConstantModel(np.zeros(K)+.1))
    beta = Node(models.ConstantModel(np.zeros(Nwords)+.01))
    thetas = [Node(models.DirichletModel(K)) for i in xrange(N)]
    graph.add_edges([alpha],thetas)

    psis = [Node(models.DirichletModel(Nwords)) for i in xrange(K)]
    graph.add_edges([beta], psis)

    Zmodel = models.CategoricalModel(K)
    Wmodel = models.ChoiceModel(models.CategoricalModel(Nwords))
    for i,doc in enumerate(documents):
        for w in doc:
            zij = Node(Zmodel)
            observed = Node(models.ConstantModel(w), Wmodel)

            graph.add_edge(thetas[i], zij)
            graph.add_edge(zij, observed)
            graph.add_edges(psis, [observed])
    return graph


