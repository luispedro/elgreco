#include <cmath>
#include <algorithm>
#include <cstring>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf_gamma.h>

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

float dirichlet_logP(float* value, float* alphas, int dim, bool normalise=true) {
    float res = 0;
    for (int i = 0; i != dim; ++i) {
        if(alphas[i]) res += alphas[i]*std::log(value[i]);
    }
    if (normalise) {
        float sumalphas = 0;
        for (int i = 0; i != dim; ++i) {
            res -= gsl_sf_lngamma(alphas[i]);
            sumalphas += alphas[i];
        }
        res += gsl_sf_lngamma(sumalphas);
    }

    return res;
}
void dirichlet_sample(random_source& R, float* res, float* alphas, int dim) {
    float V = 0.;
    for (int i = 0; i != dim; ++i) {
        res[i] = R.gamma(alphas[i], 1.0);
        V += res[i];
    }
    float min = V/dim/1000.;
    for (int i = 0; i != dim; ++i) {
        res[i] /= V;
        if (res[i] < min) res[i] = min;
    }
}

void dirichlet_sample_uniform(random_source& R, float* res, float alpha, int dim) {
    float alphas[dim];
    std::fill(alphas, alphas + dim, alpha);
    dirichlet_sample(R, res, alphas, dim);
}

float multinomial_mixture_p(float t, const float* bj, const int* cj, const int N) {
    float res = 1.;
    for (int j = 0; j != N; ++j) {
        const float bt = 1-bj[j]*t;
        res *= bt;
    }
    return std::log(res);
}
void sample_multinomial_mixture(random_source& R, float* res, const float* alphas, int dim, float** multinomials, const int* counts_idx, const int* counts) {
    float proposal[dim];
    float attempt[dim];
    std::fill(proposal, proposal + dim, .1);
    // n will be the number of elements that are != 0
    int n = 0;
    for (const int* ci = counts_idx; *ci != -1; ++ci) {
        float max = multinomials[0][*ci];
        int maxi = 0;
        for (int i = 1; i != dim; ++i) {
            if (multinomials[i][*ci] > max) {
                max = multinomials[i][*ci];
                maxi = i;
            }
        }
        proposal[maxi] += 1.;
    }
    dirichlet_sample(R, attempt, proposal, dim);

    float ratio = 1.;
    for (int i = 0; i != dim; ++i) {
        const float* m = multinomials[i];
        for (const int* ci = counts_idx, *c = counts; *ci != -1; ++ci, ++c) {
            float bj;
            if (m[*ci] == 1.) {
                bj = -10000.; // A good approximation to -infinity
            } else if (m[*ci] != .5) {
                const float r = m[*ci]/float(1.-m[*ci]);
                bj = 1 - r;
            }
            const float r = (1.-bj*proposal[i])/(1.-bj*res[i]);
            for (int ii = 0; ii < *c; ++ii) {
                ratio *= r;
            }
        }
    }
    if (ratio > 1. || R.uniform01() < ratio) {
        std::memcpy(res, attempt, sizeof(attempt));
    }

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
        sample_multinomial_mixture(R, thetas_ + i*K_, alphas, K_, multinomials_, counts_idx_[i], counts_[i]);
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

void lda::lda::forward() {
    for (int i = 0; i != N_; ++i) {
        dirichlet_sample_uniform(R, thetas_ + i*K_, alpha_, K_);
    }
    for (int i = 0; i != K_; ++i) {
        dirichlet_sample_uniform(R, multinomials_[i], beta_, Nwords_);
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

