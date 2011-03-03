import numpy as np
vocab = [line.strip() for line in file('ap/vocab.txt')]
#words = np.loadtxt('betas.txt')
words = np.loadtxt('words.txt')
Nwords = 25

for ti,ws in enumerate(words):
    print 'Topic', ti
    for w in reversed(ws.argsort()[-Nwords:]):
        print vocab[w], int(1./ws[w])
    print
    print
    print
