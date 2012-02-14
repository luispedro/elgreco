from elgreco import lda

def test_lda_empty_doc():
    data = lda.lda_data()
    data.push_back_doc(range(5), [], [])
    data.push_back_doc(range(5,10), [], [])
    data.push_back_doc(range(10), [], [])
    data.push_back_doc([], [], [])
    data.push_back_doc(range(10), [], [])

        
    params = lda.lda_parameters()
    params.alpha = .01
    params.beta = .1
    params.nr_topics = 2
    params.nr_labels = 0
    params.seed = 2

    sampler = lda.lda_uncollapsed(data, params)
    sampler.forward()
    logp = sampler.logP()
    assert logp < 0


def test_save_load():
    data = lda.lda_data()
    data.push_back_doc(range(5), [], [])
    data.push_back_doc(range(5,10), [], [])
    data.push_back_doc(range(5,10), [], [])
    data.push_back_doc(range(10), [], [])
    data.push_back_doc(range(10), [], [])

    params = lda.lda_parameters()
    params.alpha = .01
    params.beta = .1
    params.nr_topics = 2
    params.nr_labels = 0
    params.seed = 2

    sampler = lda.lda_collapsed(data, params)
    sampler.forward()
    for i in xrange(10): sampler.step()
    logp = sampler.logP()
    s = lda.save_model_to_string(sampler)
    for i in xrange(10): sampler.step()
    logp2 = sampler.logP()
    lda.load_model_from_string(sampler, s)
    logp3 = sampler.logP()
    assert logp != logp2
    assert logp == logp3
    print logp, logp2, logp3


