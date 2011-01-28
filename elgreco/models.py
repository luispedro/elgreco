# Copyright (C) 2011, Luis Pedro Coelho <lpc@cmu.edu>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
#
# LICENSE: GPLv3
'''
======
Models
======

Each Model has a mirror ModelC which is the compiled version. The methods
``compiled()`` and ``interpreted()`` go back and forth.
'''

from __future__ import division
import numpy as np
from scipy import stats


def _variable_name(node):
    return node._value.variable_name()

class Dirichlet(object):
    dtype = np.ndarray
    def __init__(self, size):
        self.size = size

    def logP(self, value, parents):
        (p,) = parents
        alphas = p.value
        return np.dot(alphas, np.log(val))

    def sample1(self, _, parents, children):
        (parents,) = parents
        alphas = parents.value.copy()
        for c in children:
            if c.model.dtype == np.ndarray:
                alphas += c.value
            elif c.model.dtype == int:
                alphas[c.value] += 1
            else:
                raise ValueError('elgreco.models.Dirichlet: Cannot handle this type')
        gammas = np.array(map(stats.gamma.rvs, alphas))
        V = stats.gamma.rvs(alphas.sum())
        return gammas/V

    def sampleforward(self, _n, parents):
        return self.sample1(_n, parents, [])
    def __str__(self):
        return 'Dirichlet()'
    __repr__ = __str__

    def compiled(self):
        return DirichletC(self.size)



class DirichletC(object):
    dtype = np.ndarray
    def __init__(self, size):
        self.size = size
    def logP(self, n, parents, result_var='result'):
        (p,) = parents
        alphas = _variable_name(p)
        yield '''
        {
            float res = 0;
            for (int i = 0; i != %(dim)s; ++i) {
                res += %(alphas)s[i]*log(%(value)s);
            }
            %(result_var)s = res;
        }
''' %   {
        'alphas' : alphas,
        'value' : _variable_name(n),
        'dim' : self.size,
        'result_var' : result_var
        }

    def sample1(self, n, parents, children):
        (p,) = parents
        alphas = _variable_name(p)
        result_var = _variable_name(n)
        yield '''
        {
            float tmp_alphas[%(dim)s];
            for (int i = 0; i != %(dim)s; ++i) {
                tmp_alphas[i] = %(alphas)s[i];
            }
        ''' %   {
                'dim' : self.size,
                'alphas' : alphas,
                }
        for c in children:
            if c.model.dtype == np.ndarray:
                yield '''
                for (int i = 0; i != %(dim)s; ++i) {
                    tmp_alphas[i] += %(cvalue)s[i];
                }
                ''' % { 'dim' : self.size, 'cvalue' : _variable_name(c) }
            elif c.model.dtype == int:
                yield '''
                ++tmp_alphas[int(%(pos)s)];
                ''' % { 'pos' : _variable_name(c) }
        yield '''
            float sum_alphas;
            for (int i = 0; i != %(dim)s; ++i) {
                %(result_var)s[i] = R.sample_gamma(tmp_alphas[i]);
                sum_alphas += tmp_alphas[i];
            }
            float V = R.sample_gamma(sum_alphas);
            for (int i = 0; i != %(dim)s; ++i) {
                %(result_var)s[i] /= V;
            }
        } ''' % {
                'dim' : self.size,
                'result_var' : result_var,
                }

    def sampleforward(self, n, parents):
        return self.codesample1(self, n, parents, [])

    def __str__(self):
        return 'DirichletC()'
    __repr__ = __str__

    def interpreted(self):
        return Dirichlet(self.size)

class Constant(object):
    def __init__(self, value, base=None):
        self.value = value
        self.base = base
        self.dtype = type(value)
        try:
            self.size = len(value)
        except:
            self.size = 1

    def logP(self, value, parents):
        if value != self.value: return float('-Inf')
        if self.base is None: return 0.
        return self.base.logP(value, parents)

    def sample1(self, _n, _p, _c):
        return self.value

    def sampleforward(self, _n, _p):
        return self.value
    def __str__(self):
        return 'Constant(%s)' % self.value

    __repr__ = __str__

    def compiled(self):
        return ConstantC(self.value, self.base)

class ConstantC(Constant):

    def logP(self, value, parents, result_var='result'):
        if self.size == 1:
            if self.base is not None:
                compute_base = self.base.logP(value, parents, result_var)
            else:
                compute_base = '%s = 0.;' % result_var
            yield '''
                {
                    float t = %(value)s;
                    if (t != %(fixed)s) %(result_var)s = -std::numeric_limits<float>::infinity();
                    else {
                        %(compute_base)s
                    }
                } ''' % {
                        'value' : value,
                        'fixed' : self.value,
                        'compute_base' : compute_base,
                        'result_var' : result_var,
                        }
        else:
            yield
            '''
            %(result_var)s = 0.;
            ''' % { 'result_var' : result_var }
            for i,v in enumerate(self.value):
                yield '''
                if (%(node)s[%(i)s] != %(value)s) {
                    %(result_var)s = -std::numeric_limits<float>::infinity();
                }''' % {
                        'node' : _variable_name(n),
                        'value' : self.value[i],
                        'result_var' : result_var,
                        'i' : i,
                }
            yield '''
                if (%(result_var)s == 0.) {
                    %(compute_base)s
                }''' % {
                        'result_var' : result_var,
                        'compute_base' : self.base.logP(value, parents, result_var),
                }

    def sample1(self, _n, _p, _c):
        yield '// constantC\n'

    def sampleforward(self, n, parents):
        if self.size == 1:
            yield '%(node)s = %(value)s;' % {
                            'node' : _variable_name(n),
                            'value' : self.value,
            }
        else:
            for i,v in enumerate(self.value):
                yield '%(node)s[%(i)s] = %(value)s;\n' % {
                        'node' : _variable_name(n),
                        'value' : self.value[i],
                        'i' : i,
                }

    def interpreted(self):
        return Constant(self.value, self.base)

class FiniteUniverse(object):
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

class FiniteUniverseC(object):
    def __init__(self, universe):
        self.universe = universe

    def sample1(self, n, parents, children):
        yield '''
        {
            float ps[%(dim)s];
            float t = 0.;
        ''' % { 'dim' : len(self.universe) }
        for i,v in enumerate(self.universe):
            yield '''
            ps[%(i)s] = 0;
            %(node)s = %(v)s;
            ''' % {
                'i' : i,
                'v' : v,
                'node' : _variable_name(n),
                }
            for code in self.logP(_variable_name(n), parents, 'ps[%s]' % i):
                yield code
            for c in children:
                for code in c.model.logP(_variable_name(c), c.children, 't'):
                    yield code
                yield 'ps[%s] += t;' % i
        yield '''
            float max_ps = -std::numeric_limits<float>::infinity();
            for (int i = 0; i != %(dim)s; ++i) {
                if (ps[i] > max_ps) max_ps = ps[i];
            }
            float sum_ps = 0.;
            for (int i = 0; i != %(dim)s; ++i) {
                ps[i] -= max_ps;
                ps[i] = exp(ps[i]);
                sum_ps += ps[i];
            }
            float r = R.uniform01();
            r *= sum_ps;
            int k = 0;
            while (k < %(dim)s) {
                if (r < ps[k]) break;
                r -= ps[k];
                ++k;
            }
        ''' %   {
                'dim' : len(self.universe),
                }
        for i,v in enumerate(self.universe):
            yield '''
            if (k == %(i)s) {
                %(result_var)s = %(v)s;
            } ''' % {
                    'i' : i,
                    'v' : v,
                    'result_var' : _variable_name(n),
            }
        yield '''
        }
        '''


class Categorical(FiniteUniverse):
    dtype = int
    size = 1
    def __init__(self, k):
        FiniteUniverse.__init__(self, np.arange(k))

    def logP(self, value, parents):
        (parents,) = parents
        alpha = parents.value
        return np.log(alpha[value]/np.sum(alpha))

    def __str__(self):
        return 'Categorical(%s)' % len(self.universe)
    __repr__ = __str__

    def compiled(self):
        return CategoricalC(len(self.universe))

class CategoricalC(FiniteUniverseC):
    dtype = int
    size = 1
    def __init__(self, k):
        FiniteUniverseC.__init__(self, np.arange(k))

    def logP(self, value, parents, result_var='result'):
        (parent,) = parents
        yield '''
        {
            float val = %(value)s;
            float sum_alphas = 0;
            for (int i = 0; i != %(dim)s; ++i) sum_alphas += %(alphas)s[i];
            %(result_var)s = log(%(alphas)s[int(val)]/sum_alphas);
        } ''' % {
                'value' : value,
                'alphas' : _variable_name(parent),
                'result_var' : result_var,
                'dim' : len(self.universe),
        }

    def interpreted(self):
        return Categorical(len(self.universe))

class Binomial(Categorical):
    dtype = bool
    def __init__(self):
        Categorical.__init__(self, 2)
    def __str__(self):
        return 'Binomial()'
    __repr__ = __str__

class Multinomial(object):
    dtype = np.ndarray
    def __init__(self, n):
        self.size = n

class Choice(object):
    def __init__(self, base):
        self.base = base
        self.dtype = self.base.dtype
        self.size = self.base.size

    def logP(self, value, parents):
        return self.base.logP(value, self._select_parent(parents))

    def _select_parent(self, parents):
        switch = parents[0].value
        parent = parents[1+switch]

    def sample1(self, n, parents, children):
        return self.base.sample1(n, [self._select_parent(parents)], children)

    def sampleforward(self, n, parents):
        return self.base.sampleforward(n, [self._select_parent(parent)])

    def __str__(self):
        return 'Choice(%s)' % self.base

    def compiled(self):
        return ChoiceC(self.base)

def ChoiceC(object):
    def __init__(self, base):
        Choice.__init__(self, base)

    def logP(self, value, parents, result_var='result'):
        p0 = parents[0]
        for i,p1 in enumerate(parents[:1]):
            yield '''
                if (%(p1)s == %(p0)s) {
                    %(compute_base)s;
                } ''' % {
                'p0' : _variable_name(p0),
                'p1' : _variable_name(p1),
                'compute_base' : self.base.logP(value, p1, result_var),
            }

    def sample1(self, n, parents, children):
        p0 = parents[0]
        for i,p1 in enumerate(parents[:1]):
            yield '''
            if (%(p1)s == %(p0)s) {
                %(compute_base)s;
            } ''' % {
                'p0' : _variable_name(p0),
                'p1' : _variable_name(p1),
                'compute_base' : self.base.sample1(value, p1, children)
            }
    def sampleforward(self, n, parents):
        p0 = parents[0]
        for i,p1 in enumerate(parents[:1]):
            yield '''
            if (%(p1)s == %(p0)s) {
                %(compute_base)s;
            } ''' % {
                'p0' : _variable_name(p0),
                'p1' : _variable_name(p1),
                'compute_base' : self.base.sampleforward(value, p1)
            }

    def __str__(self):
        return 'ChoiceC(%s)' % self.base

    def interpreted(self):
        return Choice(self.base)
