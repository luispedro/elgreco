import lda
documents = []
for line in file('ap/ap.dat'):
    tokens = line.strip().split()
    tokens = tokens[1:]
    words = []
    for tok in tokens:
        v,c = tok.split(':')
        for i in xrange(int(c)):
            words.append(int(v))
    documents.append(words)

data = lda.lda_data()
for doc in documents:
    data.push_back_doc(doc, [], [])
params = lda.lda_parameters()
params.alpha = .01
params.beta = .1
params.nr_topics = 40
params.seed = 2
params.nr_labels = 0
sampler = lda.lda_uncollapsed(data, params)
sampler.forward()

logps = [sampler.logP()]
for i in xrange(100):
    sampler.step()
    lp = (sampler.logP())
    print lp
    logps.append(sampler.logP())
