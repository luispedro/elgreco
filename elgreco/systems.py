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
    alpha = Node(models.Constant(np.zeros(K)+.1), name=r'$\alpha$')
    alpha.fix(alpha.model.value)

    beta = Node(models.Constant(np.zeros(Nwords)+.2), name=r'$\beta$')
    beta.fix(beta.model.value)

    psis = [Node(models.Dirichlet(Nwords), name=r'$\Psi_%s$' % i) for i in xrange(K)]
    graph.add_edges([beta], psis)

    Zmodel = models.Categorical(K)
    Wmodel = models.MultinomialMixture(K, Nwords)
    for i,doc in enumerate(documents):
        ti = Node(models.Dirichlet(K), name=r'$\theta_%s$' % i)
        graph.add_edge(alpha,ti)

        w = np.bincount(doc)
        if len(w) < Nwords:
            w = np.concatenate([w, np.zeros(Nwords-len(w))])
        wi = Node(Wmodel, name=r'$w_i$')
        wi.fix(w)

        graph.add_edge(ti, wi, 'z')
        graph.add_edges(psis, [wi], 'psi')
    return graph


