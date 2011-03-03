#include <cmath>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <omp.h>

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

int categorical_sample_norm(random_source& R, const floating* ps, int dim) {
    const floating val = R.uniform01();
    floating s = 0;
    for (int i = 0; i != dim; ++i) {
        s += ps[i];
        if (val < s) return i;
    }
    return dim - 1;
}

int categorical_sample_cps(random_source& R, const floating* cps, int dim) {
    floating val = R.uniform01();
    if (val < cps[0]) return 0;
    val += cps[0];
    int a = 0, b = dim;
    while ((a+1) < b) {
        int m = a + (b-a)/2;
        if (val > cps[m]) a = m;
        else b = m;
    }
    return a;
}

int categorical_sample(random_source& R, const floating* ps, int dim) {
    if (dim == 1) return 0;
    floating cps[dim];
    cps[0] = ps[0];
    for (int i = 1; i != dim; ++i) {
        cps[i] = ps[i]+ cps[i-1];
    }
    for (int i = 0; i != dim; ++i) {
        cps[i] /= cps[dim-1];
    }
    return categorical_sample_cps(R, cps, dim);
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
    floating proposal[dim];
    for (int k = 0; k != dim; ++k) proposal_prior[k] = 32.*thetas[k];
    dirichlet_sample(R, proposal, proposal_prior, dim);
    //floating proposal_reverseprior[dim];
    //for (int k = 0; k != dim; ++k) proposal_reverseprior[k] = 128.*proposal[k];

    floating logratio = dirichlet_logP_uniform(proposal, alpha, dim)
                    - dirichlet_logP_uniform(thetas, alpha, dim);
     for (const int* j = counts_idx, *cj = counts; *j != -1; ++j, ++cj) {
        floating sum_kp = 0;
        floating sum_kt = 0;
        for (int k = 0; k != dim; ++k) {
            const floating* m = multinomials[k];
            sum_kp += proposal[k] * m[*j];
            sum_kt += thetas[k] * m[*j];
         }
        logratio += (*cj) * std::log(sum_kp/sum_kt);
    }
    //logratio += dirichlet_logP(thetas, proposal_reverseprior, dim, true)
    //            - dirichlet_logP(proposal, proposal_prior, dim, true);
    if (logratio > 0. || std::log(R.uniform01()) < logratio) {
        std::memcpy(thetas, proposal, sizeof(floating)*dim);
    }
}

}

lda::lda_base::lda_base(lda_data& words, lda_parameters params)
    :R(params.seed)
    ,K_(params.nr_topics)
    ,N_(words.nr_docs())
    ,Nwords_(words.nr_terms())
    ,alpha_(params.alpha)
    ,beta_(params.beta) {
        int Nitems = 0;
        for (int i = 0; i != words.nr_docs(); ++i) {
            std::sort(words.at(i).begin(), words.at(i).end());
            if (words.at(i).back() > Nwords_) Nwords_ = words.at(i).back();
            Nitems += words.size(i);
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
lda::lda_uncollapsed::lda_uncollapsed(lda_data& words, lda_parameters params)
    :lda_base(words, params) {
        thetas_ = new floating[N_ * K_];
        multinomials_ = new floating*[K_];
        multinomials_data_ = new floating[K_ * Nwords_];
        for (int k = 0; k != K_; ++k) {
            multinomials_[k] = multinomials_data_ + k*Nwords_;
        }
    }

lda::lda_collapsed::lda_collapsed(lda_data& words, lda_parameters params)
    :lda_base(words, params) {
        z_ = new int[words.nr_words()];
        zi_ = new int*[N_+1];
        zi_[0] = z_;
        int* zinext = z_;
        for (int i = 0; i != N_; ++i) {
            for (const int* j = counts_idx_[i], *cj = counts_[i]; *j != -1; ++j, ++cj) {
                for (int cij = 0; cij != *cj; ++cij) {
                    ++zinext;
                }
            }
            zi_[i+1] = zinext;
        }
        topic_ = new int[K_];
        topic_count_ = new int*[N_];
        topic_count_[0] = new int[N_*K_];
        for (int m = 1; m != N_; ++m) {
            topic_count_[m] = topic_count_[m-1]+K_;
        }
        topic_term_ = new int*[Nwords_];
        topic_term_[0] = new int[Nwords_*K_];
        for (int t = 1; t != Nwords_; ++t) {
            topic_term_[t] = topic_term_[t-1] + K_;
        }
        topic_sum_ = new int[N_];
    }

void lda::lda_collapsed::step() {
    int* z = z_;
    for (int i = 0; i != N_; ++i) {
        for (const int* j = counts_idx_[i], *cj = counts_[i]; *j != -1; ++j, ++cj) {
            for (int cij = 0; cij != *cj; ++cij) {
                floating p[K_];
                const int ok = *z;
                --topic_count_[i][ok];
                --topic_sum_[i];
                --topic_[ok];
                --topic_term_[*j][ok];
                for (int k = 0; k != K_; ++k) {
                    p[k] = (topic_term_[*j][k] + beta_)/
                                (topic_[k] + beta_) *
                        (topic_count_[i][k] + alpha_)/
                                (topic_sum_[i] + alpha_ - 1);
                    if (k > 0) p[k] += p[k-1];
                }
                for (int k = 0; k != K_; ++k) p[k] /= p[K_-1];
                const int k = categorical_sample_cps(R, p, K_);

                *z++ = k;
                ++topic_count_[i][k];
                ++topic_sum_[i];
                ++topic_[k];
                ++topic_term_[*j][k];
            }
        }
    }
}
void lda::lda_uncollapsed::step() {
    random_source R2 = R;
    R.uniform01();
    floating proposal[K_ * Nwords_];
    std::fill(proposal, proposal + K_*Nwords_, beta_);

    #pragma omp parallel firstprivate(R2) shared(proposal)
    {
        floating* priv_proposal = new floating[K_*Nwords_];
        std::fill(priv_proposal, priv_proposal + K_*Nwords_, 0.);
        //std::cout << "step(): " << (void*)priv_proposal << '\n';
        #pragma omp for
        for (int i = 0; i < N_; ++i) {
            floating* Ti = (thetas_ + i *K_);
            sample_multinomial_mixture(R2, Ti, alpha_, K_, multinomials_, counts_idx_[i], counts_[i]);
            for (int k = 1; k != K_; ++k) Ti[k] += Ti[k-1];
            for (const int* j = counts_idx_[i], *cj = counts_[i]; *j != -1; ++j, ++cj) {
                for (int cji = 0; cji != (*cj); ++cji) {
                    ++priv_proposal[categorical_sample_cps(R2, Ti, K_)*Nwords_ + *j];
                }
            }
        }

        #pragma omp critical
        {
            floating* p = priv_proposal;
            for (int kj = 0; kj < K_*Nwords_; ++kj) proposal[kj] += *p++;
        }
        delete [] priv_proposal;

        #pragma omp barrier
        #pragma omp for nowait
        for (int k = 0; k < K_; ++k) {
            dirichlet_sample(R2, multinomials_[k], proposal+ k*Nwords_, Nwords_);
        }
        #pragma omp for
        for (int i = 0; i < N_; ++i) {
            floating* Ti = (thetas_ + i *K_);
            for (int k = K_-1; k > 0; --k) Ti[k] -= Ti[k-1];
        }

    }
}

void lda::lda_uncollapsed::forward() {
    for (int i = 0; i != N_; ++i) {
        dirichlet_sample_uniform(R, thetas_ + i*K_, alpha_, K_);
    }
    for (int i = 0; i != K_; ++i) {
        dirichlet_sample_uniform(R, multinomials_[i], beta_, Nwords_);
    }
}

void lda::lda_collapsed::forward() {
    std::fill(topic_, topic_ + K_, 0);
    int* z = z_;
    for (int i = 0; i != N_; ++i) {
        for (const int* j = counts_idx_[i], *cj = counts_[i]; *j != -1; ++j, ++cj) {
            for (int cji = 0; cji != (*cj); ++cji) {
                const int k = R.random_int(0, K_);
                *z++ = k;
                ++topic_count_[i][k];
                ++topic_sum_[i];
                ++topic_term_[*j][k];
                ++topic_[k];
            }
        }
    }
}


floating lda::lda_uncollapsed::logP(bool normalise) const {
    double p = 0.;
    #pragma omp parallel for reduction(+:p)
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

floating lda::lda_collapsed::logP(bool normalise) const {
    floating logp = 0;
    if (normalise) {
        logp += N_ * ( gsl_sf_lngamma(K_ * alpha_) - K_* gsl_sf_lngamma(alpha_));
    }
    #pragma omp parallel for reduction(+:logp)
    for (int i = 0; i < N_; ++i) {
        const int* zi = zi_[i];
        floating counts[K_];
        int count = 0;
        std::fill(counts, counts + K_, 0);
        for (const int* zi = zi_[i]; zi != zi_[i+1]; ++zi) {
            ++counts[*zi];
            ++count;
        }
        for (int k = 0; k != K_; ++k) {
            logp += gsl_sf_lngamma(counts[k] + alpha_);
        }
        logp -= gsl_sf_lngamma(count + K_*alpha_);
    }
    return logp;
}


void lda::lda_uncollapsed::print_topics(std::ostream& out) const {
    const floating* t = thetas_;
    for (int i = 0; i != N_; ++i) {
        for (int k = 0; k != K_; ++k) {
            out << *t++ << '\t';
        }
        out << '\n';
    }
}

void lda::lda_uncollapsed::print_words(std::ostream& out) const {
    for (int k = 0; k != K_; ++k) {
        floating* m = multinomials_[k];
        for (int j = 0; j != Nwords_; ++j) {
            out << m[j] << '\t';
        }
        out << '\n';

    }
}
void lda::lda_collapsed::print_words(std::ostream& out) const {
    floating multinomials[Nwords_];
    for (int k = 0; k != K_; ++k) {
        std::fill(multinomials, multinomials + Nwords_, beta_);
        const int* z = z_;
        for (int i = 0; i != N_; ++i) {
            for (const int* j = counts_idx_[i], *cj = counts_[i]; *j != -1; ++j, ++cj) {
                for (int cji = 0; cji != (*cj); ++cji) {
                    if (*z++ == k) ++multinomials[*j];
                }
            }
        }
        const floating sum_m = std::accumulate(multinomials, multinomials + Nwords_, 0.);
        for (int j = 0; j != Nwords_; ++j) {
            out << multinomials[j]/sum_m << '\t';
        }
        out << '\n';
    }
}

void lda::lda_collapsed::print_topics(std::ostream& out) const {
    floating thetas[K_];
    const int* z = z_;
    for (int i = 0; i != N_; ++i) {
        std::fill(thetas, thetas + K_, alpha_);
        for (const int* j = counts_idx_[i], *cj = counts_[i]; *j != -1; ++j, ++cj) {
            for (int cji = 0; cji != (*cj); ++cji) {
                ++thetas[*z++];
            }
        }
        const floating sum_t = std::accumulate(thetas, thetas + K_, 0.);
        for (int k = 0; k != K_; ++k) {
            out << thetas[k]/sum_t << '\t';
        }
        out << '\n';
    }
}

void lda::lda_uncollapsed::load(std::istream& topics, std::istream& words) {
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

