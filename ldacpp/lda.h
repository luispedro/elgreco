#ifndef LDA_H_INCLUDE_GUARD_LPC_ELGRECO_
#define LDA_H_INCLUDE_GUARD_LPC_ELGRECO_

#include <vector>
#include <istream>

#include "MersenneTwister.h"

namespace lda {

typedef double float_t;
typedef MTRand random_source;

struct lda_parameters {
    int nr_topics;
    int nr_iterations;
    random_source R;
    float_t alpha;
    float_t beta;
};

struct lda_data {
    public:
        int nr_words() const {
            int res = 0;
            for (int i = 0; i != docs.size(); ++i) res += size(i);
            return res;
        }
        int nr_docs() const { return docs.size(); }
        int size(int d) const { return docs[d].size(); }
        int operator()(int d, int w) const { return docs[d][w]; }
        int nr_terms() const { return nr_terms_; }
    private:
        friend lda_data load(std::istream &in);
        std::vector< std::vector<int> > docs;
        int nr_terms_;
};

struct lda_state {
    lda_state()
        :z(0)
        ,topic_count(0)
        ,topic_sum(0)
        ,topic_term(0)
        ,topic(0)
        ,alpha(0)
        ,beta(0)
        { }
    int* z;
    int** topic_count;
    int* topic_sum;
    int** topic_term;
    int* topic;
    float_t* alpha;
    float_t* beta;
};

lda_state lda(const lda_parameters& params, const lda_data& data);
lda_data load(std::istream& in);
}

#endif // LDA_H_INCLUDE_GUARD_LPC_ELGRECO_
