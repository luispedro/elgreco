import pickle
from elgreco import lda

def test_smoke():

    data = lda.lda_data()
    data.push_back_doc(range(5), [], [])
    data.push_back_doc(range(5,10), [], [])
    data.push_back_doc(range(5,10), [], [])
    data.push_back_doc(range(10), [], [])
    data.push_back_doc(range(10), [], [])

    params = lda.lda_parameters()
    params.alpha = .01
    params.beta = .1
    params.lam = .1
    params.nr_topics = 2
    params.nr_labels = 0
    params.seed = 2
    params.area_markers.push_back(1024)

    sampler = lda.lda_collapsed(data, params)
    sampler.forward()
    for i in xrange(10): sampler.step()
    from elgreco import ldahelper
    ldahelper.lda_sampler_py(sampler)
    s = ldahelper.lda_sampler_py(sampler)
    s = pickle.loads(pickle.dumps(s))
    s.get()

