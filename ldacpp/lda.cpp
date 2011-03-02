#include <cmath>
#include <algorithm>
#include <cstring>

#include "lda.h"

namespace{

floating dirichlet_logP(const floating* value, const floating* alphas, int dim, bool normalise=true) {
    floating res = 0;
    for (int i = 0; i != dim; ++i) {
        if(alphas[i] && value[i] > 0.) res += alphas[i]*std::log(value[i]);
    }
    if (normalise) {
        floating sumalphas = 0;
        for (int i = 0; i != dim; ++i) {
            res -= gsl_sf_lngamma(alphas[i]);
            sumalphas += alphas[i];
        }
        res += gsl_sf_lngamma(sumalphas);
    }

    return res;
}
floating dirichlet_logP_uniform(const floating* value, const floating alpha, int dim, bool normalise=true) {
    floating res = 0;
    for (int i = 0; i != dim; ++i) {
        if(value[i] > 0.) res += std::log(value[i]);
    }
    res *= alpha;
    if (normalise) {
        res -= dim*gsl_sf_lngamma(alpha);
        res += gsl_sf_lngamma(dim* alpha);
    }

    return res;
}
void dirichlet_sample(random_source& R, floating* res, const floating* alphas, int dim) {
    floating V = 0.;
    const floating small = dim*1.e-9;
    for (int i = 0; i != dim; ++i) {
        res[i] = R.gamma(alphas[i], 1.0);
        V += res[i];
    }
    for (int i = 0; i != dim; ++i) {
        res[i] /= V;
    }
}

void dirichlet_sample_uniform(random_source& R, floating* res, floating alpha, int dim) {
    floating alphas[dim];
    std::fill(alphas, alphas + dim, alpha);
    dirichlet_sample(R, res, alphas, dim);
}

bool binomial_sample(random_source& R, floating p) {
    return R.uniform01() < p;
}

int categorical_sample(random_source& R, const floating* ps, int dim) {
    if (dim == 1) return 0;
    floating cps[dim];
    cps[0] = ps[0];
    for (int i = 1; i != dim; ++i) {
        cps[i] += ps[i]+ cps[i-1];
    }
    const floating val = cps[dim-1] * R.uniform01();
    int a = 0, b = dim;
    while ((a+1) < b) {
        int m = a + (b-a)/2;
        if (cps[m] > val) b = m;
        else a = m;
    }
    return a;
}


void add_noise(random_source& R, floating* res, int dim, floating avg) {
    for (int i = 0; i != dim; ++i) {
        res[i] += avg * R.uniform01() - avg/2.;
        if (res[i] < 0) res[i] = -res[i];
        if (res[i] > 1) res[i] = 1-res[i];
    }
    floating ps = 0;
    for (int i = 0; i != dim; ++i) ps += res[i];
    for (int i = 0; i != dim; ++i) res[i] /= ps;
}

void sample_multinomial_mixture(random_source& R, floating* thetas, const floating alpha, int dim, floating** multinomials, const int* counts_idx, const int* counts) {
    floating proposal_prior[dim];
    for (int k = 0; k != dim; ++k) proposal_prior[k] = alpha;
    for (const int* j = counts_idx, *cj = counts; *j != -1; ++j, ++cj) {
        for (int cji = 0; cji != (*cj); ++cji) {
            floating post[dim];
            std::memcpy(post, thetas, sizeof(post));
            for (int k = 0; k != dim; ++k) {
                post[k] = multinomials[k][*j];
            }
            ++proposal_prior[categorical_sample(R, post, dim)];
        }
    }
    dirichlet_sample(R, thetas, proposal_prior, dim);
}

}

lda::lda::lda(lda_data& words, lda_parameters params)
    :R(params.seed)
    ,K_(params.nr_topics)
    ,N_(words.nr_docs())
    ,Nwords_(words.nr_terms())
    ,alpha_(params.alpha)
    ,beta_(params.beta) {
        thetas_ = new floating[N_ * K_];

        int Nitems = 0;
        for (int i = 0; i != words.nr_docs(); ++i) {
            std::sort(words.at(i).begin(), words.at(i).end());
            if (words.at(i).back() > Nwords_) Nwords_ = words.at(i).back();
            Nitems += words.size(i);
        }
        multinomials_ = new floating*[K_];
        multinomials_data_ = new floating[K_ * Nwords_];
        for (int k = 0; k != K_; ++k) {
            multinomials_[k] = multinomials_data_ + k*Nwords_;
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
            floating c = 1;
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


void lda::lda::step() {
    for (int i = 0; i < N_; ++i) {
        sample_multinomial_mixture(R, thetas_ + i*K_, alpha_, K_, multinomials_, counts_idx_[i], counts_[i]);
    }

    floating proposal[K_][Nwords_];
    for (int k = 0; k < K_; ++k) {
        for (int j = 0; j < Nwords_; ++j) proposal[k][j] = beta_;
    }
    for (int i = 0; i != N_; ++i) {
        const floating* Ti = (thetas_ + i *K_);
        for (const int* j = counts_idx_[i], *cj = counts_[i]; *j != -1; ++j, ++cj) {
            for (int cji = 0; cji != (*cj); ++cji) {
                ++proposal[categorical_sample(R, Ti, K_)][*j];
            }
        }
    }
    for (int k = 0; k < K_; ++k) {
        dirichlet_sample(R, multinomials_[k], proposal[k], Nwords_);
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


floating lda::lda::logP(bool normalise) const {
    double p = 0.;
    for (int i = 0; i < N_; ++i) {
        const floating* Ti = (thetas_ + i*K_);
        p += dirichlet_logP_uniform(Ti, alpha_, K_, normalise);
        //std::cout << p << '\n';
        // compute p += \sum_j w_j  * log( \sum_k \theta_k \psi_k )
        // we use an intermediate variable (local_p) to avoid adding really
        // small numbers to a larger number
        floating local_p = 0;
        for (const int* j = counts_idx_[i], *cj = counts_[i]; *j != -1; ++j, ++cj) {
            floating sum_k = 0.;
            for (int k = 0; k != K_; ++k) {
                sum_k += Ti[k] * multinomials_[k][*j];
            }
            //std::cout << "sum_k: " << sum_k << '\n';
            local_p += (*cj) * std::log(sum_k);
        }
        //std::cout << "local_p: " << local_p << '\n';
        p += local_p;
        //std::cout << p << '\n';
        if (normalise) {
            std::cerr << "normalise not implemented.\n";
        }
    }
    for (int k = 0; k != K_; ++k) {
        p += dirichlet_logP_uniform(multinomials_[k], beta_, Nwords_, normalise);
        //std::cout << p << '\n';
    }
    return p;
}

void lda::lda::print_topics(std::ostream& out) const {
    const floating* t = thetas_;
    for (int i = 0; i != N_; ++i) {
        for (int k = 0; k != K_; ++k) {
            out << *t++ << '\t';
        }
        out << '\n';
    }
}
void lda::lda::print_words(std::ostream& out) const {
    for (int k = 0; k != K_; ++k) {
        floating* m = multinomials_[k];
        for (int j = 0; j != Nwords_; ++j) {
            out << m[j] << '\t';
        }
        out << '\n';

    }
}

void lda::lda::load(std::istream& topics, std::istream& words) {
    floating* t = thetas_;
    for (int i = 0; i != N_*K_; ++i) {
        topics >> *t++;
    }
    if (!topics) {
        std::cerr << "Error reading topics file.\n";
    }
    for (int k = 0; k != K_; ++k) {
        floating* m = multinomials_[k];
        for (int j = 0; j != Nwords_; ++j) {
            words >> *m++;
        }
    }
    if (!words) {
        std::cerr << "Error reading words file.\n";
    }
}

