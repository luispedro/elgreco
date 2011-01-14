#ifndef LDA_H_INCLUDE_GUARD_LPC_ELGRECO_
#define LDA_H_INCLUDE_GUARD_LPC_ELGRECO_

#include <vector>

typedef double float_t;

struct random_source {
    float_t uniform01() {
        return .1;
    }
};

struct lda_parameters {
    int nr_topics;
    int nr_iterations;
    random_source R;
};

struct lda_data {
    int nr_docs() const { return docs.size(); }
    int size(int d) const { return docs[d].size(); }
    int operator()(int d, int w) const { return docs[d][w]; }
    std::vector< std::vector<int> > docs;
};

struct lda_state {
    int* z;
    int** topic_count;
    int* topic_sum;
    int** topic_term;
    int* topic;
    float_t* alpha;
    float_t* beta;
};

#endif // LDA_H_INCLUDE_GUARD_LPC_ELGRECO_
