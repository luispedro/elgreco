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

float dirichlet_logP(const float* value, const float* alphas, int dim, bool normalise=true) {
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
void dirichlet_sample(random_source& R, float* res, const float* alphas, int dim) {
    float V = 0.;
    for (int i = 0; i != dim; ++i) {
        res[i] = R.gamma(alphas[i], 1.0);
        V += res[i];
    }
    const float min = V/dim/1000.;
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

void add_noise(random_source& R, float* res, int dim, float avg) {
    float noise[dim];
    float mean = 0;
    for (int i = 0; i != dim; ++i) {
        noise[i] = avg * R.uniform01();
        mean += noise[i];
    }
    mean /= dim;
    for (int i = 0; i != dim; ++i) {
        res[i] += noise[i]-mean;
        if (res[i] < 0) res[i] = -res[i];
        if (res[i] > 1) res[i] = 1-res[i];
    }
    float ps = 0;
    for (int i = 0; i != dim; ++i) ps += res[i];
    for (int i = 0; i != dim; ++i) res[i] /= ps;
}



float multinomial_mixture_p(float t, const float* bj, const int* cj, const int N) {
    float res = 1.;
    for (int j = 0; j != N; ++j) {
        const float bt = 1-bj[j]*t;
        res *= bt;
    }
    return std::log(res);
}
float accept_logratio = 0.;
int accept = 0;
int reject = 0;
void sample_multinomial_mixture(random_source& R, float* thetas, const float* alphas, int dim, float** multinomials, const int* counts_idx, const int* counts) {
    float proposal_prior[dim];
    float proposal[dim];
    std::fill(proposal_prior, proposal_prior + dim, .1);
    for (const int* ci = counts_idx; *ci != -1; ++ci) {
        float max = multinomials[0][*ci];
        float ps = 0;
        int maxi = 0;
        for (int i = 1; i != dim; ++i) {
            ps += multinomials[i][*ci];
            if (multinomials[i][*ci] > max) {
                max = multinomials[i][*ci];
                maxi = i;
            }
        }
        if (max/ps > .8) proposal_prior[maxi] += 1;
    }
    //dirichlet_sample(R, proposal, proposal_prior, dim);
    std::memcpy(proposal, thetas, sizeof(proposal));
    add_noise(R, proposal, dim, .02);

    float logratio = 0.;
    for (const int* j = counts_idx, *cj = counts; *j != -1; ++j, ++cj) {
        float sum_kp = 0;
        float sum_kt = 0;
        for (int k = 0; k != dim; ++k) {
            const float* m = multinomials[k];
            sum_kp += proposal[k] * m[*j];
            sum_kt += thetas[k] * m[*j];
        }
        logratio += (*cj) * std::log(sum_kp/sum_kt);
    }
    //logratio += dirichlet_logP(thetas, proposal_prior, dim)
    //            - dirichlet_logP(proposal, proposal_prior, dim);
    if (logratio > 0. || std::log(R.uniform01()) < logratio) {
        accept_logratio += logratio;
        ++accept;
        std::memcpy(thetas, proposal, sizeof(float)*dim);
        //std::cout << "accept [" << logratio << "]\n";
    } else {
        ++reject;
        //std::cout << "reject [" << logratio << "]\n";
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
    accept_logratio = 0.;
    accept = 0;
    reject = 0;
    float alphas[K_];
    std::fill(alphas, alphas + K_, alpha_);
    for (int i = 0; i != N_; ++i) {
        sample_multinomial_mixture(R, thetas_ + i*K_, alphas, K_, multinomials_, counts_idx_[i], counts_[i]);
    }
    float scratchWords[Nwords_];
    for (int k = 0; k != K_; ++k) {
        float proposal[Nwords_];
        std::memcpy(proposal, multinomials_[k], sizeof(proposal));
        add_noise(R, proposal, Nwords_, .02);

        float logratio = 0.;
        for (int i = 0; i != N_; ++i) {
            const float* Ti = (thetas_ + i *K_);
            for (const int* j = counts_idx_[i], *cj = counts_[i]; *j != -1; ++j, ++cj) {
                float sum_km = 0;
                for (int k2 = 0; k2 != K_; ++k2) {
                    sum_km += Ti[k2] * multinomials_[k2][*j];
                }
                float sum_kp = sum_km;
                sum_kp += Ti[k] * (proposal[*j] - multinomials_[k][*j]);
                logratio += (*cj) * std::log(sum_kp/sum_km);
            }
        }
        //logratio += dirichlet_logP(Ti, proposal_prior, dim)
        //            - dirichlet_logP(proposal, proposal_prior, dim);
        if (logratio > 0. || std::log(R.uniform01()) < logratio) {
            accept_logratio += logratio;
            std::memcpy(multinomials_[k], proposal, sizeof(proposal));
            //std::cout << "accept [" << logratio << "]\n";
        } else {
            //std::cout << "reject [" << logratio << "]\n";
        }
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


float lda::lda::logP(bool normalise) const {
    float alphas[K_];
    float betas[Nwords_];
    std::fill(alphas, alphas + K_, alpha_);
    std::fill(betas, betas + Nwords_, beta_);
    double p = 0.;
    for (int i = 0; i != N_; ++i) {
        const float* Ti = (thetas_ + i*K_);
        float local_p = 0;
        p += dirichlet_logP(Ti, alphas, K_, normalise);
        // compute p += \sum_j w_j  * log( \sum_k \theta_k \psi_k )
        for (const int* j = counts_idx_[i], *cj = counts_[i]; *j != -1; ++j, ++cj) {
            float sum_k = 0.;
            for (int k = 0; k != K_; ++k) {
                sum_k += Ti[k] * multinomials_[k][*j];
            }
            local_p += (*cj) * std::log(sum_k);
        }
        p += local_p;
        if (normalise) {
            std::cerr << "normalise not implemented.\n";
        }
    }
    for (int k = 0; k != K_; ++k) {
        p += dirichlet_logP(multinomials_[k], betas, Nwords_, normalise);
    }
    return p;
}

