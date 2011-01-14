#include "lda.h"
#include <vector>



lda_state new_lda_state(const lda_parameters& params, const lda_data& data) {
    const int nr_docs = data.nr_docs();
    const int nr_words = data.nr_words();
    const int nr_topics = params.nr_topics;
    const int max_term = data.max_term();

    lda_state res;
    res.z = new int[nr_words];
    res.topic = new int[nr_topics];
    res.topic_count = new int*[nr_docs];
    res.topic_count[0] = new int[nr_docs*nr_topics];
    for (int m = 1; m != nr_docs; ++m) {
        res.topic_count[m] = res.topic_count[m-1]+nr_topics;
    }
    res.topic_term = new int*[max_term];
    res.topic_term[0] = new int[max_term*nr_topics];
    for (int t = 1; t != max_term; ++t) {
        res.topic_term[t] = res.topic_term[t-1] + nr_topics;
    }
    res.alpha = new float_t[nr_topics];
    res.beta = new float_t[nr_topics];
    for (int k = 0; k != nr_topics; ++k) {
        res.alpha[k] = params.alpha;
        res.beta[k] = params.beta;
    }
    return res;
}



// Implementation based on the description in "Parameter estimation for text
// analysis" by Gregor Heinrich
lda_state lda(const lda_parameters& params, const lda_data& data) {
    lda_state state = new_lda_state(params, data);
    random_source R = params.R;
    int* z = state.z;
    for (int m = 0; m != data.nr_docs(); ++m) {
        for (int k = 0; k != params.nr_topics; ++k) {
            state.topic_count[m][k] = 0;
        }
        state.topic_sum[m] = 0;
        for (int j = 0; j != data.size(m); ++j) {
            const int t = data(m,j);
            const int k = int(R.uniform01() * params.nr_topics + .5);
            *z++ = k;
            ++state.topic_count[m][k];
            ++state.topic_sum[m];
            ++state.topic_term[t][k];
            ++state.topic[k];
        }
    }
    std::vector<double> p;
    p.resize(params.nr_topics+1);
    for (int i = 0; i != params.nr_iterations; ++i) {
        z = state.z;
        for (int m = 0; m != data.nr_docs(); ++m) {
            for (int j = 0; j != data.size(m); ++j) {
                const int ok = *z;
                const int t = data(m, j);
                --state.topic_count[m][ok];
                --state.topic_sum[m];
                --state.topic[ok];
                --state.topic_term[t][ok];
                double total_p = 0;
                for (int k = 0; k != params.nr_topics; ++k) {
                    p[k] = (state.topic_term[t][k] + state.beta[k])/
                                (state.topic[k] + state.beta[k]) *
                        (state.topic_count[m][k] + state.alpha[k])/
                                (state.topic_sum[k] + state.alpha[k] - 1);
                    total_p += p[k];
                    if (k > 0) p[k] += p[k-1];
                }
                p[params.nr_topics] = p[params.nr_topics-1]+1.;
                const double s = total_p * R.uniform01();
                int k;
                while (k < params.nr_topics && s < p[k + 1]) ++k;

                *z++ = k;
                ++state.topic_count[m][k];
                ++state.topic_sum[m];
                ++state.topic[k];
                ++state.topic_term[t][k];
            }
        }
    }
    return state;
}

