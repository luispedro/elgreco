#include <cmath>
#include <algorithm>
#include <iostream>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#include "lda.h"

namespace{
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
};

random_source R = random_source(2);

float dirichlet_logP(float* value, float* alphas, int dim) {
    float res = 0;
    for (int i = 0; i != dim; ++i) {
        res += alphas[i]*std::log(value[i]);
    }
    return res;
}
void dirichlet_sample(random_source& R, float* res, float* alphas, int dim) {
    float sum_alphas = 0.;
    for (int i = 0; i != dim; ++i) {
        res[i] = R.gamma(alphas[i], 1.0);
        sum_alphas += alphas[i];
    }
    float V = R.gamma(sum_alphas, 1.0);
    for (int i = 0; i != dim; ++i) {
        res[i] /= V;
    }
}
float multinomial_mixture_p(float t, const float* bj, const int* cj, const int N, const float a0, const float a1) {
    float res = std::pow(t, a0)*std::pow(1.-t, a1);
    for (int j = 0; j != N; ++j) {
        const float bt = 1-bj[j]*t;
        res *= bt;
    }
    return res;
}
float sample_multinomial_mixture1(random_source& R, float* bj, const int* counts, const int N, const float a0, const float a1) {
    const int Nr_samples = 100;
    float ps[Nr_samples];
    float cumsum = 0;
    for (int i = 0; i != Nr_samples; ++i) {
        ps[i] = multinomial_mixture_p(float(i)/Nr_samples, bj, counts, N, a0, a1);
        cumsum += ps[i];
    }
    const float val = cumsum * R.uniform01();
    int a = 0;
    int b = Nr_samples;
    while (a + 1 < b) {
        int m = a + (b-a)/2;
        if (val < ps[m]) b = m;
        else a = m;
    }
    return float(a)/Nr_samples;
}
void sample_multinomial_mixture(random_source& R, float* res, const float* alphas, int dim, float** multinomials, const int* counts_idx, const int* counts, float* bj) {
    float rem = 1.;
    for (int i = 0; i != (dim - 1); ++i) {
        const float* c0 = multinomials[i];
        // n will be the number of elements that are != 0
        int n = 0;
        const int* c = counts;
        for (const int* ci = counts_idx; *ci != -1; ++ci, ++c) {
            if (c0[*ci] == 1.) {
                bj[n++] = -10000.; // A good approximation to -infinity
            } else if (c0[*ci] != .5) {
                const float r = c0[*ci]/float(1.-c0[*ci]);
                bj[n++] = 1 - r;
            }
            for (int ii = 1; ii < *c; ++ii) {
                bj[n] = bj[n-1];
                ++n;
            }
        }
        const float v = rem*sample_multinomial_mixture1(R, bj, counts, n, alphas[i], 1.-alphas[i]);
        res[i] = v;
        rem -= v;
    }
    res[dim - 1] = rem;
}

}

lda::lda::lda(lda_data& words, lda_parameters params)
    :K_(params.nr_topics)
    ,N_(words.nr_docs())
    ,Nwords_(words.nr_terms())
    ,alpha_(params.alpha)
    ,beta_(params.beta) {
        
        thetas_ = new float[N_ * K_];

        int Nitems = 0;
        for (int i = 0; i != words.nr_docs(); ++i) {
            std::sort(words.at(i).begin(), words.at(i).end());
            if (words.at(i).back() > Nwords_) Nwords_ = words.at(i).back();
            Nitems += words.size(i);
        }
        multinomials_ = new float*[K_];
        multinomials_data_ = new float[K_ * Nwords_];
        for (int k = 0; k != K_; ++k) {
            multinomials_[k] = multinomials_data_ + k;
        }
        counts_ = new int*[N_];
        counts_data_ = new int[Nitems]; // this is actually an overestimate, but that's fine
        counts_idx_ = new int*[N_];
        counts_idx_data_ = new int[Nitems + N_];
        
        int* j = counts_idx_data_;
        int* cj = counts_data_;
        for (int i = 0; i != N_; ++i) {
            counts_idx_[i] = j;
            counts_[i] = cj;
            float c = 1;
            int prev = words(i, 0);
            for (int ci = 1; ci < words.size(i); ++ci) {
                if (words(i, ci) != prev) {
                    *j++ = prev;
                    *cj++ = c;
                    prev = words(i, ci);
                    c = 1;
                } else {
                    ++c;
                }
            }
            *j++ = prev;
            *cj++ = c;
            *j++ = -1;
        }
    }


void lda::lda::gibbs() {
    float alphas[K_];
    float bj[N_];
    std::fill(alphas, alphas + K_, alpha_);
    // preprocess_for_theta
    for (int i = 0; i != Nwords_; ++i) {
        float ps = 0;
        for (int k = 0; k != K_; ++k) {
            ps += multinomials_[k][i];
        }
        for (int k = 0; k != K_; ++k) {
            multinomials_[k][i] /= ps;
        }
    }
    for (int i = 0; i != N_; ++i) {
        sample_multinomial_mixture(R, thetas_ + i*K_, alphas, K_, multinomials_, counts_idx_[i], counts_[i], bj);
    }
    float scratchWords[Nwords_];
    for (int k = 0; k != K_; ++k) {
        std::fill(scratchWords, scratchWords + Nwords_, beta_);
        for (int i = 0; i != N_; ++i) {
            const float weight = (thetas_ + i*K_)[k];
            for (const int* j = counts_idx_[i], *cj = counts_[i]; *j != -1; ++j, ++cj) {
                scratchWords[*j] += weight * (*cj);
            }
        }
        dirichlet_sample(R, multinomials_[k], scratchWords, Nwords_);
    }
}

float lda::lda::logP() const {
    float alphas[K_];
    float betas[Nwords_];
    std::fill(alphas, alphas + K_, alpha_);
    std::fill(betas, betas + Nwords_, beta_);
    float p = 0.;
    for (int i = 0; i != N_; ++i) {
        p += dirichlet_logP(thetas_ + i*K_, alphas, K_);
    }
    for (int k = 0; k != K_; ++k) {
        p += dirichlet_logP(multinomials_[k], betas, Nwords_);
    }
    return p;
}

