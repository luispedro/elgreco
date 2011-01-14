#ifndef LDA_H_INCLUDE_GUARD_LPC_ELGRECO_
#define LDA_H_INCLUDE_GUARD_LPC_ELGRECO_

#include <vector>
#include <istream>

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
        int max_term() const { return max_term_; }
    private:
        friend lda_data load(std::istream &in);
        std::vector< std::vector<int> > docs;
        int max_term_;
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

#endif // LDA_H_INCLUDE_GUARD_LPC_ELGRECO_
