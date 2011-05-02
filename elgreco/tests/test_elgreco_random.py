import elgreco_random

def test_normal_params():
    params = elgreco_random.normal_params(0.,1.)
    assert elgreco_random.normal_like(-1, params) == elgreco_random.normal_like(1, params)
    assert elgreco_random.normal_like(-1.3, params) == elgreco_random.normal_like(1.3, params)

def test_R_normal():
    R = elgreco_random.random_source(20)
    n01s = [R.normal(0,1) for i in xrange(100)]
    assert sum((n > 0) for n in n01s) > 33
    assert sum((n < 0) for n in n01s) > 33
    assert not sum((n < -10) for n in n01s)
    assert not sum((n > 10) for n in n01s)


def test_R_gamma():
    R = elgreco_random.random_source(20)
    assert 48 < sum(R.gamma(5,10) for i in xrange(10000))/10000.  < 52

def test_truncated_normal():
    R = elgreco_random.random_source(20)
    assert min(elgreco_random.left_truncated_normal(R, 1.) for i in xrange(1000)) > 1.
    assert min(elgreco_random.left_truncated_normal(R, 1.) for i in xrange(1000)) < 1.1
    assert -1.1 < min(elgreco_random.left_truncated_normal(R, -1.) for i in xrange(1000)) < -0.96

