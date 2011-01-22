# Copyright (C) 2011, Luis Pedro Coelho <lpc@cmu.edu>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
#
# LICENSE: GPLv3

from __future__ import division
import numpy as np
from scipy import stats

class DirichletModel(object):
    def __init__(self):
        pass

    def logP(self, value, parents):
        (p,) = parents
        alphas = p.value
        return np.dot(alphas, np.log(val))

    def sample1(self, _, parents, children):
        (parents,) = parents
        alphas = parents.value.copy()
        for c in children:
            if isinstance(c.model, MultinomialModel):
                alphas += c.value
            elif isinstance(c.model, CategoricalModel):
                alphas[c.value] += 1
            else:
                raise ValueError('elgreco.models.DirichletModel: Cannot handle this type')
        gammas = np.array(map(stats.gamma.rvs, alphas))
        V = stats.gamma.rvs(alphas.sum())
        return gammas/V

    def sampleforward(self, _n, parents):
        return self.sample1(_n, parents, [])

    def __str__(self):
        return 'DirichletModel()'
    __repr__ = __str__

class ConstantModel(object):
    def __init__(self, value, base=None):
        self.value = value
        self.base = base

    def logP(self, value, parents):
        if value != self.value: return float('-Inf')
        if self.base is None: return 0.
        return self.base.logP(value, parents)

    def sample1(self, _n, _p, _c):
        return self.value

    def sampleforward(self, _n, _p):
        return self.value
    def __str__(self):
        return 'ConstantModel(%s)' % self.value
    __repr__ = __str__

class FiniteUniverseModel(object):
    def __init__(self, universe):
        self.universe = universe

    def sample1(self, n, parents, children):
        ps = []
        for v in self.universe:
            n.value = v
            cur = n.logP()
            for c in children:
                cur += c.logP()
            ps.append(cur)
        ps = np.array(ps)
        ps -= ps.max()
        np.exp(ps,ps)
        ps /= ps.sum()
        r = np.random.random()
        v = 0
        ps[-1] += 1.
        pc = ps[0]
        while pc < r:
            v += 1
            pc += ps[v]
        return self.universe[v]

    def sampleforward(self, n, parents):
        return self.sample1(n, parents, [])

class CategoricalModel(FiniteUniverseModel):
    def __init__(self, k):
        FiniteUniverseModel.__init__(self, np.arange(k))

    def logP(self, value, parents):
        (parents,) = parents
        alpha = parents.value
        return np.log(alpha[value]/np.sum(alpha))

    def __str__(self):
        return 'CategoricalModel(%s)' % len(self.universe)
    __repr__ = __str__

class BinomialModel(CategoricalModel):
    def __init__(self):
        CategoricalModel.__init__(self, 2)
    def __str__(self):
        return 'BinomialModel()'
    __repr__ = __str__

class MultinomialModel(object):
    def __init__(self):
        pass

