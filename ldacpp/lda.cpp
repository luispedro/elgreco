#include <cmath>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <limits>

#include <omp.h>

#include "lda.h"

namespace{

const floating _pi = 3.1415926535898;
const floating _inv_two_pi = 1./std::sqrt(2. * _pi);

floating normal_like(const floating value, const lda::normal_params& params, bool normalise=true) {
    floating d = (value - params.mu) * std::sqrt(params.precision);
    return _inv_two_pi * params.precision * std::exp( -d*d );
}

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
floating logdirichlet_logP_uniform(const floating* value, const floating alpha, int dim, bool normalise=true) {
    floating res = 0;
    for (int i = 0; i != dim; ++i) {
        res += value[i];
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
    for (int i = 0; i != dim; ++i) {
        res[i] = R.gamma(alphas[i], 1.0);
        V += res[i];
    }
    for (int i = 0; i != dim; ++i) {
        res[i] /= V;
    }
}
void logdirichlet_sample(random_source& R, floating* res, const floating* alphas, int dim) {
    floating V = 0.;
    for (int i = 0; i != dim; ++i) {
        if (alphas[i] < 1) {
            const floating u = R.uniform01();
            const floating v = R.gamma(alphas[i]+1.0, 1.);
            res[i] = std::log(v) + std::log(u)/alphas[i];
            V += v*std::pow(u, 1./alphas[i]);
        } else {
            const floating v = R.gamma(alphas[i], 1.0);
            res[i] = std::log(v);
            V += v;
        }
    }
    V = std::log(V);
    for (int i = 0; i != dim; ++i) {
        res[i] -= V;
    }
}

void dirichlet_sample_uniform(random_source& R, floating* res, floating alpha, int dim) {
    floating alphas[dim];
    std::fill(alphas, alphas + dim, alpha);
    dirichlet_sample(R, res, alphas, dim);
}

void logdirichlet_sample_uniform(random_source& R, floating* res, floating alpha, int dim) {
    floating alphas[dim];
    std::fill(alphas, alphas + dim, alpha);
    logdirichlet_sample(R, res, alphas, dim);
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
    int a = 0, b = dim;
    while ((a+1) < b) {
        const int m = a + (b-a)/2;
        if (val < cps[m]) b = m;
        else a = m;
    }
    return a + 1;
}

inline
void ps_to_cps(floating* ps, int dim) {
    for (int i = 1; i != dim; ++i) ps[i] += ps[i-1];
    if (ps[dim-1] == 0) {
        for (int i = 0; i != dim; ++i) ps[i] = i/(dim-1);
    } else {
        for (int i = 0; i != dim; ++i) ps[i] /= ps[dim-1];
    }
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

}

lda::lda_base::lda_base(lda_data& words, lda_parameters params)
    :R(params.seed)
    ,K_(params.nr_topics)
    ,N_(words.nr_docs())
    ,F_(words.nr_features())
    ,Nwords_(words.nr_terms())
    ,alpha_(params.alpha)
    ,beta_(params.beta)
    ,Ga_(1)
    ,Gb_(1)
    ,Gn0_(1)
    ,Gmu_(0.)
    {
        int Nitems = 0;
        for (int i = 0; i != words.nr_docs(); ++i) {
            std::sort(words.at(i).begin(), words.at(i).end());
            if (words.at(i).back() > Nwords_) Nwords_ = words.at(i).back();
            Nitems += words.size(i);
        }
        counts_ = new int*[N_];
        counts_[0] = new int[Nitems]; // this is actually an overestimate, but that's fine
        counts_idx_ = new int*[N_];
        counts_idx_[0] = new int[Nitems + N_];

        features_ = new floating*[N_];
        features_[0] = new floating[N_*F_];
        for (int i = 1; i != N_; ++i) {
            features_[i] = features_[i-1] + F_;
        }
        for (int i = 0; i != N_; ++i) {
            for (int f = 0; f != F_; ++f) {
                features_[i][f] = words.feature(i,f);
            }
        }
        int* j = counts_idx_[0];
        int* cj = counts_[0];
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
        multinomials_[0] = new floating[K_ * Nwords_];
        for (int k = 1; k != K_; ++k) {
            multinomials_[k] = multinomials_[k-1] + Nwords_;
        }
        normals_ = new normal_params*[K_];
        normals_[0] = new normal_params[K_*F_];
        for (int k = 1; k != K_; ++k) {
            normals_[k] = normals_[k-1] + F_;
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
    floating crossed[K_ * Nwords_];
    std::fill(proposal, proposal + K_*Nwords_, beta_);
    floating f_sum[K_*F_];
    floating f_sum2[K_*F_];
    std::fill(f_sum, f_sum + K_*F_, 0.);
    std::fill(f_sum2, f_sum2 + K_*F_, 0.);

    #pragma omp parallel firstprivate(R2) shared(proposal, crossed)
    {
        floating* priv_proposal = new floating[K_*Nwords_];
        floating* f_priv = new floating[K_*F_];
        floating* f_priv2 = new floating[K_*F_];
        std::fill(priv_proposal, priv_proposal + K_*Nwords_, 0.);
        std::fill(f_priv, f_priv + K_*F_, 0.);
        std::fill(f_priv2, f_priv2 + K_*F_, 0.);

        #pragma omp for
        for (int j = 0; j < Nwords_; ++j) {
            floating* crossed_j = crossed + j*K_;
            floating max = multinomials_[0][j];
            for (int k = 0; k != K_; ++k) {
                crossed_j[k] = multinomials_[k][j];
                if (crossed_j[k] > max) max = crossed_j[k];
            }
            for (int k = 0; k != K_; ++k) {
                crossed_j[k] = std::exp(crossed_j[k] - max);
            }
        }

        #pragma omp for
        for (int i = 0; i < N_; ++i) {
            floating* Ti = (thetas_ + i *K_);
            floating Tp[K_];
            std::fill(Tp, Tp + K_, alpha_);
            floating p[K_];
            for (const int* j = counts_idx_[i], *cj = counts_[i]; *j != -1; ++j, ++cj) {
                floating* crossed_j = crossed + (*j)*K_;
                for (int k = 0; k != K_; ++k) {
                    p[k] = Ti[k]*crossed_j[k];
                }
                ps_to_cps(p, K_);
                for (int cji = 0; cji != (*cj); ++cji) {
                    int z = categorical_sample_cps(R2, p, K_);
                    ++priv_proposal[z*Nwords_ + *j];
                    ++Tp[z];
                }
            }
            for (int f = 0; f != F_; ++f) {
                for (int k = 0; k != K_; ++k) {
                    p[k] = Ti[k]*normal_like(features_[i][f], normals_[k][f]);
                }
                ps_to_cps(p, K_);
                const int z = categorical_sample_cps(R2, p, K_);
                floating fif = features_[i][f];
                int kf = f*K_ + z;
                f_priv[kf] += fif;
                f_priv2[kf] += fif*fif;
            }
            dirichlet_sample(R2, Ti, Tp, K_);
        }

        #pragma omp critical
        {
            floating* p = priv_proposal;
            for (int kj = 0; kj < K_*Nwords_; ++kj) proposal[kj] += *p++;
            for (int kf = 0; kf < K_*F_; ++kf) {
                f_sum[kf] += f_priv[kf];
                f_sum2[kf] += f_priv2[kf];
            }
        }
        delete [] priv_proposal;
        delete [] f_priv;
        delete [] f_priv2;

        #pragma omp barrier
        #pragma omp for nowait
        for (int k = 0; k < K_; ++k) {
            logdirichlet_sample(R2, multinomials_[k], proposal+ k*Nwords_, Nwords_);
            for (int f = 0; f < F_; ++f) {
                // Check:
                // 'http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.126.4603&rep=rep1&type=pdf'
                // or
                // http://en.wikipedia.org/wiki/Normal-gamma_distribution#Posterior_distribution_of_the_parameters
                const floating sum = f_sum[f*K_ + k];
                const floating sum2 = f_sum2[f*K_ + k];
                const floating x_bar = sum/N_;
                const floating tao = R.gamma(Ga_ + N_/2., Gb_ + (sum2 - 2*sum*x_bar + x_bar*x_bar)/2. + N_*Gn0_/2./(N_+Gn0_)*(x_bar-Gmu_)*(x_bar-Gmu_));
                const floating mu = R.normal( N_*tao/(N_*tao + Gn0_ * tao)*x_bar + Gn0_*tao/(N_*tao+Gn0_*tao)*Gmu_, (N_+Gn0_)*tao);
                normals_[k][f] = normal_params(mu, tao);
            }
        }
    }
}

void lda::lda_uncollapsed::forward() {
    for (int i = 0; i != N_; ++i) {
        dirichlet_sample_uniform(R, thetas_ + i*K_, alpha_, K_);
    }
    for (int k = 0; k != K_; ++k) {
        logdirichlet_sample_uniform(R, multinomials_[k], beta_, Nwords_);
        for (int f = 0; f < F_; ++f) {
            const floating tao = R.gamma(Ga_, Gb_);
            const floating mu = R.normal(Gn0_* Gmu_, Gn0_*tao);
            normals_[k][f] = normal_params(mu, tao);
        }
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
    floating crossed[K_ * Nwords_];

    #pragma omp parallel shared(crossed)
    {
        #pragma omp for
        for (int j = 0; j < Nwords_; ++j) {
            floating* crossed_j = crossed + j*K_;
            floating max = crossed_j[0];
            for (int k = 0; k != K_; ++k) {
                crossed_j[k] = multinomials_[k][j];
                if (crossed_j[k] > max) max = crossed_j[k];
            }
            for (int k = 0; k != K_; ++k) {
                crossed_j[k] = std::exp(crossed_j[k] - max);
            }
        }

        #pragma omp for reduction(+:p)
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
                floating* crossed_j = crossed + (*j)*K_;
                for (int k = 0; k != K_; ++k) {
                    sum_k += Ti[k] * crossed_j[k];
                }
                local_p += (*cj) * std::log(sum_k);
            }
            p += local_p;
            if (normalise) {
                std::cerr << "normalise not implemented.\n";
            }
        }
        #pragma omp for reduction(+:p)
        for (int k = 0; k < K_; ++k) {
            p += logdirichlet_logP_uniform(multinomials_[k], beta_, Nwords_, normalise);
        }
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

int lda::lda_uncollapsed::retrieve_logbeta(int k, float* res, int size) const {
    if (size != Nwords_) return 0;
    for (int j = 0; j != Nwords_; ++j) {
        res[j] = multinomials_[k][j];
    }
    return Nwords_;
}
int lda::lda_uncollapsed::retrieve_theta(int i, float* res, int size) const {
    if (size != K_) return 0;
    const floating* m = thetas_ + i*K_;
    for (int k = 0; k != K_; ++k) res[k] = m[k];
    return K_;
}

void lda::lda_uncollapsed::load(std::istream& topics, std::istream& words) {
    floating* t = thetas_;
    for (int i = 0; i != N_*K_; ++i) {
        topics >> *t++;
    }
    if (!topics) {
        std::cerr << "Error reading topics file.\n";
    }
    for (int i = 0; i != N_; ++i) {
        const floating sum = std::accumulate(thetas_ + i *K_, thetas_ + (i+1)*K_, 0.);
        if (sum < .95 || sum > 1.05) {
            std::cerr << "Malformed topics file (entry: " << i << ")\n";
        }
    }

    floating* m = multinomials_[0];
    for (int kj = 0; kj != Nwords_*K_; ++kj) {
        words >> *m++;
    }
    if (!words) {
        std::cerr << "Error reading words file.\n";
    }
    for (int k = 0; k != K_; ++k) {
        const floating sum = std::accumulate(multinomials_[k], multinomials_[k] + Nwords_, 0.);
        if (sum < .95 || sum > 1.05) {
            std::cerr << "Malformed words file (entry: " << k << ")\n";
        }
    }
}


