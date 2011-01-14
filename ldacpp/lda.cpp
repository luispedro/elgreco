#include "lda.h"
#include <vector>


// Implementation based on the description in "Parameter estimation for text
// analysis" by Gregor Heinrich
int lda(const lda_parameters& params, const lda_data& data, lda_state* state) {
    random_source R = params.R;
    int* z = state->z;
    for (int m = 0; m != data.nr_docs; ++m) {
        for (int k = 0; k != params.nr_topics; ++k) {
            state->topic_count[m][k] = 0;
        }
        state->topic_sum[m] = 0;
        for (int j = 0; j != data.lengths[m]; ++j) {
            const int t = data.words[m][j];
            const int k = int(R.uniform01() * params.nr_topics + .5);
            *z++ = k;
            ++state->topic_count[m][k];
            ++state->topic_sum[m];
            ++state->topic_term[t][k];
            ++state->topic[k];
        }
    }
    std::vector<double> p;
    p.resize(params.nr_topics+1);
    for (int i = 0; i != params.nr_iterations; ++i) {
        z = state->z;
        for (int m = 0; m != data.nr_docs; ++m) {
            for (int j = 0; j != data.lengths[m]; ++j) {
                const int ok = *z;
                const int t = data.words[m][j];
                --state->topic_count[m][ok];
                --state->topic_sum[m];
                --state->topic[ok];
                --state->topic_term[t][ok];
                double total_p = 0;
                for (int k = 0; k != params.nr_topics; ++k) {
                    p[k] = (state->topic_term[t][k] + state->beta[k])/
                                (state->topic[k] + state->beta[k]) *
                        (state->topic_count[m][k] + state->alpha[k])/
                                (state->topic_sum[k] + state->alpha[k] - 1);
                    total_p += p[k];
                    if (k > 0) p[k] += p[k-1];
                }
                p[params.nr_topics] = p[params.nr_topics-1]+1.;
                const double s = total_p * R.uniform01();
                int k;
                while (k < params.nr_topics && s < p[k + 1]) ++k;

                *z++ = k;
                ++state->topic_count[m][k];
                ++state->topic_sum[m];
                ++state->topic[k];
                ++state->topic_term[t][k];
            }
        }
    }
}

