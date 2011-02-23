#ifndef LDA_H_INCLUDE_GUARD_LPC_ELGRECO_
#define LDA_H_INCLUDE_GUARD_LPC_ELGRECO_
#include <vector>
#include <iostream>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf_gamma.h>

struct random_source {
    random_source(unsigned s)
        :r(gsl_rng_alloc(gsl_rng_mt19937))
    {
        gsl_rng_set(r, s);
    }
    ~random_source() {
        gsl_rng_free(r);
    }


    float uniform01() {
        return gsl_ran_flat(r, 0., 1.);
    }
    float gamma(float a, float b) {
        return gsl_ran_gamma(r, a, b);
    }
    private:
       gsl_rng * r;
       random_source(const random_source&);
       random_source& operator=(const random_source&);
};


namespace lda {
struct lda_parameters {
    unsigned seed;
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
        float logP(bool normalise=false) const;

    private:
        random_source R;
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
