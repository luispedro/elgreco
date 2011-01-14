#ifndef LDA_H_INCLUDE_GUARD_LPC_ELGRECO_
#define LDA_H_INCLUDE_GUARD_LPC_ELGRECO_
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
    const int nr_docs;
    const int* lengths;
    const int** words;
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
