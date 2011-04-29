#include <cmath>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <limits>
#include <assert.h>

#include <omp.h>

#include "lda.h"

#include <gsl/gsl_linalg.h>

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
    assert(val < cps[dim-1]);
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
    std::copy(ps, ps + dim, cps);
    ps_to_cps(cps, dim);
    return categorical_sample_cps(R, cps, dim);
}

template<typename F1, typename F2>
floating dot_product(const F1* x, const F2* y, const int dim) {
    floating res = 0;
    for (int i = 0; i != dim; ++i) res += x[i] * y[i];
    return res;
}

floating left_truncated_normal(random_source& R, const floating mu) {
    if (mu <= 0) {
        floating z;
        do {
            z = R.normal(0., 1.);
        } while (z < mu);
        return z;
    } else {
        const floating alphastar = (mu + std::sqrt(mu*mu + 4))/2.;
        int iters = 0;
        while (true) {
            const floating z = alphastar + R.exponential(1./alphastar);
            const floating rho = std::exp(-(z-alphastar)*(z-alphastar)/2.);
            const floating u = R.uniform01();
            if (u < rho) return z;
            ++iters;
            if (iters > 100) {
                std::cerr << ">100 iters for " << mu << '\n';
            }
        }
    }
}

}

lda::lda_base::lda_base(lda_data& input, lda_parameters params)
    :R(params.seed)
    ,K_(params.nr_topics)
    ,N_(input.nr_docs())
    ,F_(input.nr_features())
    ,L_(params.nr_labels)
    ,Nwords_(input.nr_terms())
    ,alpha_(params.alpha)
    ,beta_(params.beta)
    ,Ga_(1)
    ,Gb_(1)
    ,Gn0_(1)
    ,Gmu_(0.)
    {
        int Nitems = 0;
        for (int i = 0; i != input.nr_docs(); ++i) {
            std::sort(input.at(i).begin(), input.at(i).end());
            if (input.at(i).back() > Nwords_) Nwords_ = input.at(i).back();
            Nitems += input.size(i);
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
                features_[i][f] = input.feature(i,f);
            }
        }
        if (params.nr_labels) {
            ls_ = new floating[N_ * L_];
            for (int i = 0; i != N_; ++i) {
                for (int ell = 0; ell != L_; ++ell) {
                    ls(i)[ell] = input.label(i, ell);
                }
            }
        } else {
            ls_ = 0;
        }
        int* j = counts_idx_[0];
        int* cj = counts_[0];
        for (int i = 0; i != N_; ++i) {
            counts_idx_[i] = j;
            counts_[i] = cj;
            floating c = 1;
            int prev = input(i, 0);
            for (int ci = 1; ci < input.size(i); ++ci) {
                if (input(i, ci) != prev) {
                    *j++ = prev;
                    *cj++ = int(c);
                    prev = input(i, ci);
                    c = 1;
                } else {
                    ++c;
                }
            }
            *j++ = prev;
            *cj++ = int(c);
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
        sample_ = new bool[N_];
        std::fill(sample_, sample_ + N_, true);

        z_bars_ = new floating[N_ * K_];
        std::fill(z_bars_, z_bars_ + N_*K_, 0.);

        if (ls_) {
            zs_ = new int*[N_];
            for (int i = 0; i != N_; ++i) {
                zs_[i] = new int[words.size(i) + F_ + 1];
                zs_[i][0] = words.size(i) + F_;
            }
            ys_ = new floating[N_*L_];
            gamma_ = new floating[K_ * L_];
            std::fill(gamma_, gamma_ + K_* L_, 0.);
            for (int i = 0; i != N_; ++i) {
                for (int ell = 0; ell != L_; ++ell) {
                floating* yi = ys(i);
                    yi[ell] = left_truncated_normal(R, 0);
                    if (!ls(i)[ell]) yi[ell] = -yi[ell];
                }
            }
        } else {
            zs_ = 0;
            ys_ = 0;
            gamma_ = 0;
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
            floating Tp[K_];
            floating* zb = z_bar(i);
            std::fill(Tp, Tp + K_, alpha_);
            floating p[K_];
            floating transfer[K_][K_];
            int wj = 1;
            const int Ni = (ls_ ? zs_[i][0] : 1);

            if (ls_) {
                for (int k = 0; k != K_; ++k) {
                    std::copy(thetas(i), thetas(i+1), transfer[k]);
                }
                for (int ell = 0; ell != L_; ++ell) {
                    const floating* gl = gamma(ell);
                    const floating delta = ys(i)[ell] - dot_product(z_bar(i), gl, K_);
                    for (int cur = 0; cur != K_; ++cur) {
                        for (int k = 0; k != K_; ++k) {
                            floating d2 = delta + (gl[cur]-gl[k])/Ni;
                            d2 *= d2;
                            transfer[cur][k] *= std::exp(-.5 * d2);
                        }
                    }
                }
            } else {
                std::copy(thetas(i), thetas(i+1), transfer[0]);
            }

            for (const int* j = counts_idx_[i], *cj = counts_[i]; *j != -1; ++j, ++cj) {
                floating* crossed_j = crossed + (*j)*K_;
                for (int cji = 0; cji != (*cj); ++cji) {
                    const int cur = (ls_ ? zs_[i][wj] : 0);
                    assert (cur < K_);
                    for (int k = 0; k != K_; ++k) {
                        p[k] = crossed_j[k] * transfer[cur][k];
                    }
                    const int z  = categorical_sample(R2, p, K_);
                    if (ls_) zs_[i][wj++] = z;
                    assert(*j < Nwords_);
                    assert(z < K_);
                    ++priv_proposal[z*Nwords_ + *j];
                    ++Tp[z];
                }
            }
            for (int f = 0; f != F_; ++f) {
                for (int k = 0; k != K_; ++k) {
                    const int cur = (ls_ ? zs_[i][wj] : 0);
                    assert(cur < K_);
                    p[k] = transfer[cur][k]*normal_like(features_[i][f], normals_[k][f]);
                }
                const int z = categorical_sample(R2, p, K_);
                if (ls_) zs_[i][wj++] = z;
                ++Tp[z];
                floating fif = features_[i][f];
                int kf = f*K_ + z;
                f_priv[kf] += fif;
                f_priv2[kf] += fif*fif;
            }
            if (sample_[i]) {
                dirichlet_sample(R2, thetas(i), Tp, K_);
            }
            std::copy(Tp, Tp + K_, zb);
            for (int k = 0; k != K_; ++k) {
                zb[k] /= Ni;
            }
            if (ls_) {
                floating* li = ls(i);
                floating* yi = ys(i);
                for (int ell = 0; ell != L_; ++ell) {
                    floating mu = dot_product(zb, gamma(ell), K_);
                    if (!li[ell]) mu = -mu;
                    yi[ell] = mu + left_truncated_normal(R2, -mu);
                    if (!li[ell]) yi[ell] = -yi[ell];
                    //std::cerr << "mu_i: " << mu << "; ys_i: " << yi[ell] << " (ls_[i]: " << floating(ls_[i]) << ")\n";
                }
            }
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
        #pragma omp for
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
        if (ls_) {
            #pragma omp for
            for (int ell = 0; ell < L_; ++ell) {
                // sample gamma
                double* zdata = new double[N_ * K_];
                std::copy(z_bars_, z_bars_ + N_ * K_, zdata);
                gsl_matrix Z;
                Z.size1 = N_;
                Z.size2 = K_;
                Z.tda = K_;
                Z.data = zdata;
                Z.block = 0;
                Z.owner = 0;

                gsl_vector b;
                b.size = N_;
                b.stride = L_;
                b.data = ys_ + ell;
                b.block = 0;
                b.owner = 0;

                gsl_vector gammav;
                gammav.size = K_;
                gammav.stride = 1;
                gammav.data = gamma(ell);
                gammav.block = 0;
                gammav.owner = 0;

                gsl_vector* r = gsl_vector_alloc(N_);
                gsl_vector* tau = gsl_vector_alloc(K_);

                gsl_linalg_QR_decomp(&Z, tau);
                gsl_linalg_QR_lssolve(&Z, tau, &b, &gammav, r);

                gsl_vector_free(tau);
                gsl_vector_free(r);
                delete [] zdata;
            }
        }
    }
}

void lda::lda_uncollapsed::forward() {
    for (int i = 0; i != N_; ++i) {
        dirichlet_sample_uniform(R, thetas(i), alpha_, K_);
        floating tcps[K_];
        std::copy(thetas(i), thetas(i+1), tcps);
        ps_to_cps(tcps, K_);
        if (ls_) {
            const int Ni = zs_[i][0];
            for (int j = 0; j != Ni; ++j) {
                zs_[i][j+1] = categorical_sample_cps(R, tcps, K_);
            }
        }
    }
    for (int k = 0; k != K_; ++k) {
        logdirichlet_sample_uniform(R, multinomials_[k], beta_, Nwords_);
        for (int f = 0; f < F_; ++f) {
            const floating tao = R.gamma(Ga_, Gb_);
            const floating mu = R.normal(Gn0_* Gmu_, Gn0_*tao);
            normals_[k][f] = normal_params(mu, tao);
        }
    }
    if (gamma_) std::fill(gamma_, gamma_ + K_ * L_, 0.);
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
    floating offset[Nwords_];

    #pragma omp parallel shared(crossed)
    {
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
            offset[j] = max;
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
                local_p += (*cj) * (std::log(sum_k) + offset[*j]);
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
    std::copy(multinomials_[k], multinomials_[k] + Nwords_, res);
    return Nwords_;
}
int lda::lda_uncollapsed::retrieve_theta(int i, float* res, int size) const {
    if (size != K_) return 0;
    const floating* m = thetas_ + i*K_;
    std::copy(m, m + K_, res);
    return K_;
}
int lda::lda_uncollapsed::retrieve_gamma(int ell, float* res, int size) const {
    if (size != K_) return 0;
    std::copy(gamma(ell), gamma(ell + 1), res);
    return K_;
}
int lda::lda_uncollapsed::retrieve_ys(int i, float* res, int size) const {
    if (size != L_) return 0;
    std::copy(ys(i), ys(i+1), res);
    return L_;
}


int lda::lda_uncollapsed::set_logbeta(int k, float* src, int size) {
    if (size != Nwords_) return 0;
    std::copy(src, src + Nwords_, multinomials_[k]);
    return Nwords_;
}
int lda::lda_uncollapsed::set_theta(int i, float* src, int size) {
    if (size != K_) return 0;
    floating* m = thetas_ + i*K_;
    std::copy(src, src + K_, m);
    return K_;
}

int lda::lda_uncollapsed::project_one(const std::vector<int>& words, float* res, int size) {
    if (size != K_) return 0;
    floating thetas[K_];
    this->sample_one(words, thetas);
    std::copy(thetas, thetas + K_, res);
    return size;
}

float lda::lda_uncollapsed::score_one(int ell, const float* res, int size) const {
    if (size != K_) return 0;
    return dot_product(res, gamma(ell), size);
}

floating lda::lda_uncollapsed::logperplexity(const std::vector<int>& words) {
    floating thetas[K_];
    this->sample_one(words, thetas);
    floating crossed[K_ * Nwords_];
    floating offset[Nwords_];

    for (int j = 0; j < Nwords_; ++j) {
        floating* crossed_j = crossed + j*K_;
        floating max = multinomials_[0][j];
        for (int k = 0; k != K_; ++k) {
            crossed_j[k] = multinomials_[k][j];
            if (crossed_j[k] > max) max = crossed_j[k];
        }
        offset[j] = max;
        for (int k = 0; k != K_; ++k) {
            crossed_j[k] = std::exp(crossed_j[k] - max);
        }
    }
    floating logp = dirichlet_logP_uniform(thetas, alpha_, K_, false);
    for (int j = 0; j != words.size(); ++j) {
        floating sum_k = 0;
        floating* crossed_j = crossed + words[j]*K_;
        for (int k = 0; k != K_; ++k) {
            sum_k += thetas[k] * crossed_j[k];
        }
        logp += std::log(sum_k) + offset[j];
    }
    return logp;
}

void lda::lda_uncollapsed::sample_one(const std::vector<int>& words, floating* thetas) {
    const int nr_iters = 20;
    std::fill(thetas, thetas + K_, 1.);

    floating crossed[K_ * Nwords_];
    floating offset[Nwords_];
    const int docsize = words.size();

    for (int j = 0; j < Nwords_; ++j) {
        floating* crossed_j = crossed + j*K_;
        floating max = multinomials_[0][j];
        for (int k = 0; k != K_; ++k) {
            crossed_j[k] = multinomials_[k][j];
            if (crossed_j[k] > max) max = crossed_j[k];
        }
        offset[j] = max;
        for (int k = 0; k != K_; ++k) {
            crossed_j[k] = std::exp(crossed_j[k] - max);
        }
    }
    for (int it = 0; it != nr_iters; ++it) {
        floating proposal[K_];
        std::fill(proposal, proposal + K_, alpha_);
        for (int j = 0; j < docsize; ++j) {
            floating p[K_];
            floating* crossed_j = crossed + words[j]*K_;
            for (int k = 0; k != K_; ++k) {
                p[k] = thetas[k]*crossed_j[k];
            }
            int z = categorical_sample(R, p, K_);
            ++proposal[z];
        }
        dirichlet_sample(R, thetas, proposal, K_);
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



