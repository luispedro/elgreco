#ifndef LDA_H_INCLUDE_GUARD_LPC_ELGRECO_
#define LDA_H_INCLUDE_GUARD_LPC_ELGRECO_
#include <vector>
#include <iostream>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf_gamma.h>

typedef double floating;

struct random_source {
    random_source(unsigned s=1234567890)
        :r(gsl_rng_alloc(gsl_rng_mt19937))
    {
        gsl_rng_set(r, s);
    }
    random_source(const random_source& rhs)
        :r(gsl_rng_clone(rhs.r))
        {
        }
    ~random_source() {
        gsl_rng_free(r);
    }


    random_source& operator = (const random_source& rhs) {
        random_source n(rhs);
        this->swap(n);
        return *this;
    }

    void swap(random_source& rhs) {
        std::swap(r, rhs.r);
    }

    floating uniform01() {
        return gsl_ran_flat(r, 0., 1.);
    }
    floating gamma(floating a, floating b) {
        return gsl_ran_gamma(r, a, b);
    }
    int random_int(int a, int b) {
        return int(gsl_ran_flat(r, double(a),double(b)));
    }
    private:
       gsl_rng * r;
};


namespace lda {
struct lda_parameters {
    unsigned seed;
    int nr_topics;
    int nr_iterations;
    floating alpha;
    floating beta;
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

struct lda_base {
    public:
        lda_base(lda_data& data, lda_parameters params);
        ~lda_base() {
            delete [] counts_;
            delete [] counts_data_;
            delete [] counts_idx_;
            delete [] counts_idx_data_;
        }
        virtual void step() = 0;
        virtual void forward() = 0;
        virtual floating logP(bool normalise=false) const = 0;

    protected:
        random_source R;
        int K_;
        int N_;
        int Nwords_;

        int** counts_;
        int* counts_data_;
        int** counts_idx_;
        int* counts_idx_data_;

        floating alpha_;
        floating beta_;
};
struct lda_uncollapsed : lda_base {
    public:
        lda_uncollapsed(lda_data& data, lda_parameters params);
        ~lda_uncollapsed() {
            delete [] multinomials_data_;
            delete [] multinomials_;
            delete [] thetas_;
        }
        virtual void step();
        virtual void forward();
        virtual floating logP(bool normalise=false) const;
        void load(std::istream& topics, std::istream& words);

        void print_topics(std::ostream&) const;
        void print_words(std::ostream&) const;

    private:
        floating** multinomials_;
        floating* multinomials_data_;
        floating* thetas_;
};

struct lda_collapsed : lda_base {
    public:
        lda_collapsed(lda_data& data, lda_parameters params);
        ~lda_collapsed() {
            delete [] z_;
            delete [] zi_;
            delete [] topic_;
            delete [] topic_count_;
            delete [] topic_sum_;
            delete [] topic_term_;
        }
        virtual void step();
        virtual void forward();
        virtual floating logP(bool normalise=false) const;

    private:
        int* z_;
        int** zi_;
        int* topic_;
        int** topic_count_;
        int* topic_sum_;
        int** topic_term_;
};

}
#endif // LDA_H_INCLUDE_GUARD_LPC_ELGRECO_
