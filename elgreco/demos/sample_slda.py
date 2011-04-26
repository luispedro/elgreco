import numpy as np
import lda

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

labels = [int(line.strip()) for line in file('train-label.dat')]

data = lda.lda_data()
for doc,lab in zip(load_data('train-data.dat'), labels):
    data.push_back_doc(doc, [], lab > 4)
    
params = lda.lda_parameters()
params.alpha = .01
params.beta = .1
params.nr_topics = 40
params.seed = 2
sampler = lda.lda_uncollapsed(data, params)
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
sampler.retrieve_gamma(G)
T = np.zeros(40, np.float32)
tests = []
for doc in tdocuments:
    _ = sampler.sample_one(doc, T)
    tests.append(sampler.score_one(T))


