import numpy as np
from time import time as now
from elgreco import lda

nr_topics = 40
nr_labels = 8
nr_f = 0

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
labs = np.zeros(nr_labels, bool)
for doc,lab in zip(load_data('../train-data.dat'), labels):
    labs[:] = 0
    labs[lab] = 1
    fs = [np.random.random() for f in xrange(nr_f)]
    data.push_back_doc(doc, fs, labs)
    
params = lda.lda_parameters()
params.alpha = .01
params.beta = .1
params.nr_topics = nr_topics
params.nr_labels = nr_labels
params.seed = 2
sampler = lda.lda_collapsed(data, params)
sampler.forward()

logps = [sampler.logP()]
#start = now()
for i in xrange(100):
    sampler.step()
    lp = (sampler.logP())
    print lp
    logps.append(lp)
#end = now()
#print end-start

tdocuments = load_data('../test-data.dat')
tlabels = [int(line.strip()) for line in file('../test-label.dat')]
G = np.zeros(nr_topics, np.float32)
sampler.retrieve_gamma(0,G)
T = np.zeros((len(tdocuments),nr_topics), np.float32)
tests = []
for ti,doc in enumerate(tdocuments):
    fs = [np.random.random() for f in xrange(nr_f)]
    s = sampler.project_one(doc, fs, T[ti])
    if s <= 0:
        raise RuntimeError('something bad happened')

    tests.append([sampler.score_one(ell,T[ti]) for ell in xrange(nr_labels)])
tests = np.array(tests)
print np.mean(tlabels == tests.argmax(1))

