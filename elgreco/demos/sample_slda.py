import numpy as np
from elgreco import lda

def load_data(datafile):
    documents = []
    for line in file(datafile):
        tokens = line.strip().split()
        tokens = tokens[1:]
        words = []
        for tok in tokens:
            v,c = tok.split(':')
            for i in xrange(int(c)):
                words.append(int(v))
        documents.append(words)
    return documents

labels = [int(line.strip()) for line in file('../train-label.dat')]

data = lda.lda_data()
labs = np.zeros(8, bool)
for doc,lab in zip(load_data('../train-data.dat'), labels):
    labs[:] = 0
    labs[lab] = 1
    fs = [np.random.random() for f in xrange(4)]
    data.push_back_doc(doc, fs, labs)
    
params = lda.lda_parameters()
params.alpha = .01
params.beta = .1
params.nr_topics = 40
params.nr_labels = 8
params.seed = 2
sampler = lda.lda_collapsed(data, params)
sampler.forward()

logps = [sampler.logP()]
for i in xrange(100):
    sampler.step()
    lp = (sampler.logP())
    print lp
    logps.append(sampler.logP())

tdocuments = load_data('test-data.dat')
tlabels = [int(line.strip()) for line in file('test-label.dat')]
G = np.zeros(40, np.float32)
sampler.retrieve_gamma(0,G)
T = np.zeros(40, np.float32)
tests = []
for doc in tdocuments:
    fs = [np.random.random() for f in xrange(4)]
    _ = sampler.project_one(doc, fs, T)
    tests.append([sampler.score_one(ell,T) for ell in xrange(8)])


