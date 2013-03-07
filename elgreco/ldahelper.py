from lda import lda_parameters, lda_data, lda_collapsed
class lda_parameters_py(object):
    def __init__(self, p):
        self.seed = p.seed
        self.nr_topics = p.nr_topics
        self.nr_labels = p.nr_labels
        self.alpha = p.alpha
        self.beta = p.beta
        self.lam = p.lam
        self.area_markers = map(int,p.area_markers)

    def get(self):
        p = lda_parameters()
        p.seed = self.seed
        p.nr_topics = self.nr_topics
        p.nr_labels = self.nr_labels
        p.alpha = self.alpha
        p.beta = self.beta
        p.lam = self.lam
        for a in self.area_markers:
            p.area_markers.push_back(int(a))
        return p

class lda_data_py(object):
    def __init__(self, d):
        self.docs = []
        for i in xrange(d.nr_docs()):
            self.docs.append(
                    (map(int,d.at(i))
                    ,map(float,d.features_at(i))
                    ,map(float,d.labels_at(i))))
                    


    def get(self):
        r = lda_data()
        for nd,nf,nl in self.docs:
            r.push_back_doc(nd,nf,nl)
        return r

class lda_sampler_py(object):
    def __init__(self, s):
        from lda import save_model_to_string
        self.data = lda_data_py(s.get_data())
        self.parameters = lda_parameters_py(s.get_parameters())
        self.state = save_model_to_string(s)

    def get(self):
        from lda import load_model_from_string
        s = lda_collapsed(self.data.get(), self.parameters.get())
        load_model_from_string(s, self.state)
        return s

