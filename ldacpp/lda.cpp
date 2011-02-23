#include <cmath>
#include <algorithm>
#include <cstring>

#include "lda.h"

namespace{

float dirichlet_logP(const float* value, const float* alphas, int dim, bool normalise=true) {
    float res = 0;
    for (int i = 0; i != dim; ++i) {
        if(alphas[i] && value[i] > 0.) res += alphas[i]*std::log(value[i]);
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
    const float small = dim*1.e-9;
    for (int i = 0; i != dim; ++i) {
        res[i] = R.gamma(alphas[i], 1.0);
        V += res[i];
    }
    for (int i = 0; i != dim; ++i) {
        res[i] /= V;
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
void sample_multinomial_mixture(random_source& R, float* thetas, const float* alphas, int dim, float** multinomials, const int* counts_idx, const int* counts) {
    float proposal_prior[dim];
    float proposal[dim];
    for (int k = 0; k != dim; ++k) proposal_prior[k] = 128.*thetas[k];
    dirichlet_sample(R, proposal, proposal_prior, dim);
    //float proposal_reverseprior[dim];
    //for (int k = 0; k != dim; ++k) proposal_reverseprior[k] = 128.*proposal[k];

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
    //logratio += dirichlet_logP(thetas, proposal_reverseprior, dim, true)
    //            - dirichlet_logP(proposal, proposal_prior, dim, true);
    if (logratio > 0. || std::log(R.uniform01()) < logratio) {
        std::memcpy(thetas, proposal, sizeof(float)*dim);
    }
}

}

lda::lda::lda(lda_data& words, lda_parameters params)
    :R(params.seed)
    ,K_(params.nr_topics)
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
        if (logratio > 0. || std::log(R.uniform01()) < logratio) {
            accept_logratio += logratio;
            std::memcpy(multinomials_[k], proposal, sizeof(proposal));
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
        p += dirichlet_logP(Ti, alphas, K_, normalise);
        // compute p += \sum_j w_j  * log( \sum_k \theta_k \psi_k )
        // we use an intermediate variable (local_p) to avoid adding really
        // small numbers to a larger number
        float local_p = 0;
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

