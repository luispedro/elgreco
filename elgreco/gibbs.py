# Copyright (C) 2011, Luis Pedro Coelho <lpc@cmu.edu>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
# 
# LICENSE: GPLv3

import numpy as np

def collect_values(g):
    '''
    values = collect_values(g)

    Parameters
    ----------
    g : Graph
        Input graph

    Returns
    -------
    values : list of values
    '''
    return [n.value for n in g.vertices]

def sampleforward(graph, R=None):
    '''
    sampleforward(graph, R=None)

    Sample one value for every node in a Bayesian network

    Parameters
    ----------
    graph : Graph
        Must be a DAG
    R : integer, optional
        Random source
    '''
    if R is not None:
        np.random.seed(R)
    queue = graph.roots()
    seen = set(queue)
    while queue:
        n = queue.pop()
        n.sampleforward()
        for c in n.children:
            queue.append(c)
            seen.add(c)

def gibbs(graph, niters, initialise=True, R=None, burn_in=0, sample_period=1, sample_function=collect_values):
    '''
    samples = gibbs(graph, niters, initialise=True, R=None, burn_in=0, sample_period=1, sample_function=collect_values):

    Gibbs Sampling

    Parameters
    ----------
    graph : Graph
        Must either be a DAG or ``initialise`` must be False
    niters : integer
        Nr of iterations
    initialise : boolean, optional
        Whether to initialise the graph (the default). If not, then the nodes
        must have their ``value`` field set. See also ``sampleforward``
    R : integer, optional
        Random seed
    burn_in : integer, optional
        Nr of iterations to ignore before collecting statistics
    sample_period : integer, optional
        Nr of iterations between statistics collection (default: 1)
    sample_function : callable, optional
        Statistics function, takes a graph and returns anything

    Returns
    -------
    values : list
        List of collected statistics
    '''
    if R is not None:
        np.random.seed(R)
    if initialise:
        sampleforward(graph, None)
    samples = []
    for i in xrange(niters):
        for n in graph.vertices:
            n.sample1()
        if i >= burn_in and (i % sample_period) == 0:
            samples.append(sample_function(graph))
    return samples

