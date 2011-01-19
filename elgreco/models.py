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

    def logP(self, value, alphas):
        return np.dot(alphas, np.log(val))

    def sample1(self, parents, children):
        (parents,) = parents
        children_stats = np.sum(children, axis=0)
        children_stats += parents
        gammas = np.array(map(stats.gamma.rvs, children_stats))
        V = stats.gamma.rvs(children_stats.sum())
        return gammas/V

class ConstantModel(object):
    def __init__(self, value):
        self.value = value

    def logP(self, value, parents):
        if value == self.value: return 0.
        return float("-Inf")

    def sample1(self, _, __):
        return self.value
