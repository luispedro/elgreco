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
    distribution = 'dirichlet'
    def __init__(self, size):
        self.size = size

    def check(self, n, parents, children):
        if children:
            if len(set([c.model.distribution for c in children])) > 1:
                raise ValueError
            if children[0].model.distribution not in (
                            'multinomial',
                            'mixture_multinomial',
                            'categorical'):
                raise ValueError
        if len(parents) != 1 or parents[0].size != self.size:
            raise ValueError

    def logP(self, value, parents):
        (p,) = parents
        alphas = p.value
        return np.dot(alphas, np.log(val))


    def __str__(self):
        return 'Dirichlet()'
    __repr__ = __str__

    def compiled(self):
        return DirichletC(self.size)

    def sample1(self, n, parents, children):
        (parents,) = parents
        alphas = parents.value.copy()
        if len(children):
            c = children[0]
            if c.model.distribution == 'multinomial':
                for c in children:
                    alphas += c.value
            elif c.model.distribution == 'mixture_multinomial':
                if [n] == c.namedparents['z']:
                    psis = c.namedparents['psi']
                    rem = 1.
                    res = []
                    for i in xrange(len(psis)-1):
                        psi0 = psis[i].value
                        a0 = alphas[i]
                        a1 = np.sum(alphas)-a0
                        psi1 = np.sum([p.value for j,p in enumerate(psis) if j != i], axis=0)
                        bj = []
                        nj = []
                        for j,cj in enumerate(c.value):
                            if cj == 0.: continue
                            rj = psi0[j]/psi1[j]
                            bj.append(1.-rj)
                            nj.append(cj)
                        bj = np.array(bj)
                        nj = np.array(nj)
                        p = np.vectorize(lambda t: np.product((1-bj*t)**nj)*t**a0*(1-t)**a1)

                        thetas = np.linspace(0, 1, 100)
                        pthetas = p(thetas)
                        r = np.sum(pthetas)*np.random.random()
                        a,b = 0, len(pthetas)
                        while a + 1 < b:
                            m = a+(b-a)//2
                            if r < pthetas[m]: b = m
                            else: a = m
                        v = rem*a/100.
                        rem -= v
                        res.append(v)
                    res.append(rem)
                    return np.array(res)
                elif n in c.namedparents['psi']:
                    alphas *= len(children)
                    for c in children:
                        (z,) = c.namedparents['z']
                        index = c.namedparents['psi'].index(n)
                        z = z.value[index]
                        alphas += z*c.value
                else:
                    raise NotImplementedError(
                        'elgreco.models.Dirichlet.sample1: Cannot handle this structure')
            elif c.model.distribution == 'categorical':
                for c in children:
                    alphas[int(c.value)] += 1
            else:
                raise ValueError('elgreco.models.Dirichlet: I cannot handle the case where my children are of type %s' % c.model.distribution)
        gammas = np.array(map(stats.gamma.rvs, alphas))
        V = stats.gamma.rvs(alphas.sum())
        return gammas/V

    def sampleforward(self, _n, parents):
        return self.sample1(_n, parents, [])


class DirichletC(Dirichlet):
    def headers(self):
        yield '''
        float dirichlet_logP(float* value, float* alphas, int dim) {
            float res = 0;
            for (int i = 0; i != dim; ++i) {
                res += alphas[i]*log(value[i]);
            }
            return res;
        }
        void dirichlet_sample(random_source& R, float* res, float* alphas, int dim) {
            float sum_alphas = 0.;
            for (int i = 0; i != dim; ++i) {
                res[i] = R.gamma(alphas[i], 1.0);
                sum_alphas += alphas[i];
            }
            float V = R.gamma(sum_alphas, 1.0);
            for (int i = 0; i != dim; ++i) {
                res[i] /= V;
            }
        }
        float multinomial_mixture_p(float t, const float* bj, const float* cj, const int N, const float alpha) {
            float res = std::pow(t, alpha)*std::pow(1.-t, alpha);
            for (int j = 0; j != N; ++j) {
                res *= std::pow(1-bj[j]*t, cj[j]);
            }
            return res;
        }
        float sample_multinomial_mixture1(random_source& R, float* bj, const float* counts, const int N, const float alpha) {
            const int Nr_samples = 100;
            float ps[Nr_samples];
            float cumsum = 0;
            for (int i = 0; i != Nr_samples; ++i) {
                ps[i] = multinomial_mixture_p(float(i)/Nr_samples, bj, counts, N, alpha);
                cumsum += ps[i];
            }
            const float val = cumsum * R.uniform01();
            int a = 0;
            int b = Nr_samples;
            while (a + 1 < b) {
                int m = a + (b-a)/2;
                if (val < ps[m]) b = m;
                else a = m;
            }
            return float(a)/Nr_samples;
        }
        void sample_multinomial_mixture(random_source& R, float* res, float* alphas, int dim, float** multinomials, float* counts, int N) {
            float rem = 1.;
            float* bj = new float[N];
            for (int i = 0; i != (dim - 1); ++i) {
                float* c0 = multinomials[i];
                for (int j = 0; j != N; ++j) {
                    float c1j = 0;
                    for (int ii = 0; ii != dim; ++ii) {
                        if (i == ii) continue;
                        c1j += multinomials[ii][j];
                    }
                    const float b = c0[j]/c1j;
                    bj[j] = 1 - b;
                }
                const float v = rem*sample_multinomial_mixture1(R, bj, counts, N, alphas[0]);
                res[i] = v;
                rem -= v;
            }
            res[dim - 1] = rem;
        }
        '''
    def logP(self, n, parents, result_var='result'):
        (p,) = parents
        alphas = _variable_name(p)
        yield '''
        %(result_var)s = dirichlet_logP(%(value)s, %(alphas)s, %(dim)s);
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
            std::memcpy(tmp_alphas, %(alphas)s, sizeof(tmp_alphas));
        ''' %   {
                'dim' : self.size,
                'alphas' : alphas,
        }
        dirichlet_sample = '''
            dirichlet_sample(R, %(result_var)s, tmp_alphas, %(dim)s);
        } ''' % {
                'dim' : self.size,
                'result_var' : result_var,
            }
        if not children:
            yield dirichlet_sample
            return
        c = children[0]
        if c.model.distribution == 'multinomial':
            for c in children:
                yield '''
                for (int i = 0; i != %(dim)s; ++i) {
                    tmp_alphas[i] += %(cvalue)s[i];
                }
                ''' % { 'dim' : self.size, 'cvalue' : _variable_name(c) }
            yield dirichlet_sample
        elif c.model.distribution == 'categorical':
            for c in children:
                yield '''
                ++tmp_alphas[int(%(pos)s)];''' % { 'pos' : _variable_name(c) }
            yield dirichlet_sample
        elif c.model.distribution == 'mixture_multinomial':
            if [n] == c.namedparents['z']:
                assert len(children) == 1
                yield '''
                    float * multinomials[%(dim)s];''' % { 'dim' : self.size, }

                for i,p in enumerate(c.namedparents['psi']):
                    yield '''
                    multinomials[%(i)s] = %(p)s;''' % {
                            'i' : i,
                            'p' : _variable_name(p)
                    }
                assert len(c.namedparents['psi']) == self.size
                yield '''
                    sample_multinomial_mixture(R, %(result_var)s, tmp_alphas, %(dim)s, multinomials, %(counts)s, %(N)s);
                } ''' % {
                    'dim' : self.size,
                    'result_var' : result_var,
                    'counts' : _variable_name(c),
                    'N' : len(c.namedparents['psi'][0].value),
               }
            elif n in c.namedparents['psi']:
                yield '''
                for (int i = 0; i != %(dim)s; ++i) tmp_alphas[i] *= %(n)s;
                ''' % { 'dim' : self.size, 'n' : len(children) }
                for c in children:
                    (z,) = c.namedparents['z']
                    index = c.namedparents['psi'].index(n)
                    yield '''
                        for (int i = 0; i != %(dim)s; ++i) {
                            tmp_alphas[i] += %(z)s[%(index)s] * %(cvalue)s[i];
                        } ''' % {
                            'dim' : self.size,
                            'z' : _variable_name(z),
                            'index' : index,
                            'cvalue' : _variable_name(c)
                        }
                yield dirichlet_sample
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    def __str__(self):
        return 'DirichletC()'

    def interpreted(self):
        return Dirichlet_Multinomial(self.size)

class Constant(object):
    def __init__(self, value):
        self.value = value
        self.dtype = type(value)
        try:
            self.size = len(value)
        except:
            self.size = 1

    def logP(self, value, parents):
        if value != self.value: return float('-Inf')
        return 0.

    def sample1(self, _n, _p, _c):
        return self.value

    def sampleforward(self, _n, _p):
        return self.value
    def __str__(self):
        return 'Constant(%s)' % self.value

    __repr__ = __str__

    def compiled(self):
        return ConstantC(self.value)

class ConstantC(Constant):

    def headers(self):
        return ()

    def logP(self, value, parents, result_var='result'):
        if self.size == 1:
            yield '''
                %(result_var)s = (%(value)s == %(fixed)s) ? 0. : -std::numeric_limits<float>::infinity();
                ''' % {
                        'value' : value,
                        'fixed' : self.value,
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
        return Constant(self.value)

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

    def headers(self):
        yield '''
        float exparray(float* ps, int dim) {
            float max_ps = -std::numeric_limits<float>::infinity();
            for (int i = 0; i != dim; ++i) {
                if (ps[i] > max_ps) max_ps = ps[i];
            }
            float sum_ps = 0.;
            for (int i = 0; i != dim; ++i) {
                ps[i] -= max_ps;
                ps[i] = exp(ps[i]);
                sum_ps += ps[i];
            }
            return sum_ps;
        }
        int samplefromlogps(random_source& R, float* ps, int dim) {
            float sum_ps = exparray(ps, dim);
            float r = R.uniform01();
            r *= sum_ps;
            int k = 0;
            while (k < dim) {
                if (r < ps[k]) break;
                r -= ps[k];
                ++k;
            }
            return k;
        }
        '''
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
            int k = samplefromlogps(R, ps, %(dim)s);
        ''' %   { 'dim' : len(self.universe), }
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
    distribution = 'categorical'
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
    distribution = 'categorical'
    def __init__(self, k):
        FiniteUniverseC.__init__(self, np.arange(k))

    def headers(self):
        for h in FiniteUniverseC.headers(self):
            yield h
        yield '''
        float categorical_logP(float val, float* alphas, int dim) {
            float sum_alphas = 0;
            for (int i = 0; i != dim; ++i) {
                sum_alphas += alphas[i];
            }
            return log(alphas[int(val)]/sum_alphas);
        }
        '''

    def logP(self, value, parents, result_var='result'):
        (parent,) = parents
        yield '''
        %(result_var)s = categorical_logP(%(value)s, %(alphas)s, %(dim)s);\
        ''' % {
                'value' : value,
                'alphas' : _variable_name(parent),
                'result_var' : result_var,
                'dim' : len(self.universe),
        }

    def sample1(self, n, parents, children):
        yield '''
        {
            float ps[%(dim)s];
            float t = 0.;
            for (int vi = 0; vi != %(k)s; ++vi) {
                ps[vi] = 0;
                %(node)s = vi;

        ''' % { 'dim' : len(self.universe),
                'k' : len(self.universe),
                'node' : _variable_name(n),
        }
        for code in self.logP(_variable_name(n), parents, 'ps[vi]'):
            yield code
        for c in children:
            for code in c.model.logP(_variable_name(c), c.children, 't'):
                yield code
            yield 'ps[vi] += t;'
        yield '''
            }
            %(result_var)s = samplefromlogps(R, ps, %(dim)s);
        } ''' %   {
                'dim' : len(self.universe),
                'result_var' : _variable_name(n),
                }
    def interpreted(self):
        return Categorical(len(self.universe))

class Binomial(Categorical):
    dtype = bool
    distribution = 'binomial'
    def __init__(self):
        Categorical.__init__(self, 2)
    def __str__(self):
        return 'Binomial()'
    __repr__ = __str__

class Multinomial(object):
    dtype = np.ndarray
    distribution = 'multinomial'
    def __init__(self, n):
        self.size = n

class MultinomialMixture(object):
    dtype = np.ndarray
    distribution = 'mixture_multinomial'
    def __init__(self, k, n):
        self.k = k
        self.size = n

    def logP(self, value, parents):
        z = parents[0]
        assert len(parents) == len(z)+1
        alphas = np.zeros(self.size)
        for zi,w in zip(z, parents[1:]):
            alphas += zi*w.value
        return np.dot(log(value), alphas)

    def sample1(self, value, parents, children):
        if len(children) != []:
            raise NotImplementedError
        raise NotImplementedError

    def sampleforward(self, value, parents):
        return self.sample1(value, parents, [])

    def __str__(self):
        return 'MultinomialMixture(%s, %s)' % (self.k, self.n)
    __repr__ = __str__

    def compiled(self):
        return MultinomialMixtureC(self.k, self.size)

class MultinomialMixtureC(MultinomialMixture):

    def headers(self):
        return ()

    def logP(self, value, parents):
        raise NotImplementedError

    def sample1(self, value, parents, children):
        raise NotImplementedError

    def sampleforward(self, value, parents):
        raise NotImplementedError

    def __str__(self):
        return 'MultinomialMixtureC(%s, %s)' % (self.k, self.n)
    __repr__ = __str__

class Choice(object):
    def __init__(self, base):
        self.base = base
        self.dtype = self.base.dtype
        self.size = self.base.size

    def logP(self, value, parents):
        return self.base.logP(value, self._select_parents(parents))

    def _select_parents(self, parents):
        switch = parents[0].value
        parent = parents[1+int(switch)]
        return [parent]

    def sample1(self, n, parents, children):
        return self.base.sample1(n, self._select_parents(parents), children)

    def sampleforward(self, n, parents):
        return self.base.sampleforward(n, self._select_parents(parents))

    def __str__(self):
        return 'Choice(%s)' % self.base

    def compiled(self):
        return ChoiceC(self.base)

def ChoiceC(object):
    def __init__(self, base):
        Choice.__init__(self, base)

    def headers(self):
        return ()

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
