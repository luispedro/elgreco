#ifndef LDA_H_INCLUDE_GUARD_LPC_ELGRECO_
#define LDA_H_INCLUDE_GUARD_LPC_ELGRECO_
#include <vector>
#include <iostream>



namespace lda {
struct lda_parameters {
    int nr_topics;
    int nr_iterations;
    float alpha;
    float beta;
};



struct lda_data {
    public:
        int nr_words() const {
            int res = 0;
            for (int i = 0; i != docs.size(); ++i) res += size(i);
            return res;
        }
        std::vector<int>& at(int d) { return docs[d]; }
        int nr_docs() const { return docs.size(); }
        int size(int d) const { return docs[d].size(); }
        int operator()(int d, int w) const { return docs[d][w]; }
        int nr_terms() const { return nr_terms_; }
    private:
        friend lda_data load(std::istream &in);
        std::vector< std::vector<int> > docs;
        int nr_terms_;
};

lda_data load(std::istream &in);

struct lda {
    public:
        lda(lda_data& data, lda_parameters params);
        ~lda() {
            delete [] counts_;
            delete [] counts_data_;
            delete [] counts_idx_;
            delete [] counts_idx_data_;
            delete [] multinomials_;
            delete [] multinomials_data_;
            delete [] thetas_;
        }
        void gibbs();
        void forward();
        float logP() const;

    private:
        int K_;
        int N_;
        int Nwords_;

        int** counts_;
        int* counts_data_;
        int** counts_idx_;
        int* counts_idx_data_;

        float alpha_;
        float beta_;
        float** multinomials_;
        float* multinomials_data_;
        float* thetas_;
};

}
#endif // LDA_H_INCLUDE_GUARD_LPC_ELGRECO_
