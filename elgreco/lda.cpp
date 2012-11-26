#include <cmath>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <limits>
#include <sstream>
#include <assert.h>
#include <boost/scoped_array.hpp>
#include <boost/scoped_ptr.hpp>

#include <omp.h>

#include "lda.h"

#include <gsl/gsl_linalg.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_sf.h>

namespace{

floating truncated_normal_like(const floating val, const floating label) {
    if (label == 0) return 1;
    if (label > 0) {
        if (val > 0) {
            return gsl_cdf_ugaussian_Q(-val);
        }
        return gsl_cdf_ugaussian_P(-val);
    }
    if (val > 0) {
        return gsl_cdf_ugaussian_P(-val);
    }
    return gsl_cdf_ugaussian_Q(-val);
}

inline
int roundi(floating x) { return int(round(x)); }

floating dirichlet_logP(const floating* value, const floating* alphas, int dim, bool normalise=true) {
    floating res = 0;
    for (int i = 0; i != dim; ++i) {
        if(alphas[i] && value[i] > 0.) res += (alphas[i]-1)*std::log(value[i]);
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
    res *= (alpha - 1);
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
    res *= (alpha - 1);
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

int categorical_sample_cps(random_source& R, const floating* cps, int dim) {
    floating val = R.uniform01();
    if (val < cps[0]) return 0;
    assert(!std::isnan(cps[dim-1]));
    assert(val <= cps[dim-1]);
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
        const floating dim_inv = 1./(dim-1);
        for (int i = 0; i != dim; ++i) ps[i] = i*dim_inv;
    } else {
        const floating ps_inv = 1./ps[dim-1];
        for (int i = 0; i != dim; ++i) ps[i] *= ps_inv;
    }
    assert(!std::isnan(ps[dim-1]));
}

int categorical_sample(random_source& R, const floating* ps, int dim) {
    if (dim == 1) return 0;
    floating cps[dim];
    std::copy(ps, ps + dim, cps);
    ps_to_cps(cps, dim);
    return categorical_sample_cps(R, cps, dim);
}

int categorical_sample_logps(random_source& R, const floating* logps, int dim) {
    floating p[dim];
    const floating* max = std::max_element(logps, logps + dim);
    for (int i = 0; i != dim; ++i) {
        p[i] = std::exp(logps[i] - *max);
    }
    return categorical_sample(R, p, dim);
}

template<typename F1, typename F2>
floating dot_product(const F1* x, const F2* y, const int dim) {
    floating res = 0;
    for (int i = 0; i != dim; ++i) res += x[i] * y[i];
    return res;
}

floating phi2(const double x) {
    // The values below were obtained by least squares fitting:
    const double x2 = x*x;
    const double x3 = x2*x;
    const double x4 = x2*x2;
    return x*0.04911332 + x2*(-0.11018848) + x3*(0.17388619) + x4*(-0.03410691);
}

floating phi(const double x) {
    if (x >= 2.0) return .997;
    if (x <= -2.0) return .003;
    if (x < 0.) return phi2(2+x);
    return 1.-phi2(2.-x);
}
bool has_diagonal_zero(gsl_matrix* mat, const int n) {
    for (int i = 0; i != n; ++i) {
        if (gsl_matrix_get(mat, i, i) == 0) {
            return true;
        }
    }
    return false;
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
    ,lambda_(params.lam)
    ,Ga_(1)
    ,Gb_(1)
    ,Gn0_(1)
    ,Gmu_(0.)
    {
        if (K_ <= 0) {
            throw "elgreco.lda: K_ must be greater than zero";
        }
        if (alpha_ <= 0) {
            throw "elgreco.lda: alpha must be strictly positive";
        }
        if (lambda_ <= 0) {
            throw "elgreco.lda: lambda must be strictly positive";
        }
        if (beta_ <= 0) {
            throw "elgreco.lda: beta must be strictly positive";
        }
        int Nitems = 0;
        for (int i = 0; i != input.nr_docs(); ++i) {
            if (!input.at(i).empty()) {
                std::sort(input.at(i).begin(), input.at(i).end());
                if (input.at(i).back() > Nwords_) Nwords_ = input.at(i).back();
                Nitems += input.size(i);
            }
        }
        words_ = new sparse_int*[N_];
        words_[0] = new sparse_int[Nitems+N_]; // this is actually an overestimate, but that's fine

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
        sparse_int* j = words_[0];
        for (int i = 0; i != N_; ++i) {
            words_[i] = j;
            int ci = 0;
            while (ci < input.size(i)) {
                const int anchor = input(i, ci++);
                int c = 1;
                while (ci < input.size(i) && input(i, ci) == anchor) {
                    ++ci;
                    ++c;
                }
                *j++ = sparse_int(anchor, c);
            }
            *j++ = sparse_int(-1, -1);
        }
        gamma_ = new floating[K_ * L_];
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
            std::fill(gamma_, gamma_ + K_* L_, 0.);
            for (int i = 0; i != N_; ++i) {
                for (int ell = 0; ell != L_; ++ell) {
                    floating* yi = ys(i);
                    yi[ell] = left_truncated_normal(R, 0);
                    if (ls(i)[ell] < 0) yi[ell] = -yi[ell];
                }
            }
        } else {
            zs_ = 0;
            ys_ = 0;
            gamma_ = 0;
        }

    }

lda::lda_collapsed::lda_collapsed(lda_data& words, lda_parameters params)
    :lda_base(words, params)
    ,area_markers_(params.area_markers)
    ,nr_areas_(params.area_markers.size()+1) {
        if (Nwords_ > area_markers_.back()) {
            throw "Nwords_ > area_markers_.back()";
        }
        zi_ = new int*[N_+1];
        zi_[0] = new int[words.nr_words() + N_*F_];
        int* zinext = zi_[0];
        for (int i = 0; i != N_; ++i) {
            int c = 0;
            for (const sparse_int* j = words_[i]; j->value != -1; ++j) {
                for (int cji = 0; cji != j->count; ++cji) {
                    ++zinext;
                    ++c;
                }
            }
            zinext += F_;
            zi_[i+1] = zinext;
        }

        topic_ = new floating[K_];
        topic_count_ = new floating*[N_];
        topic_count_[0] = new floating[N_*K_];
        for (int m = 1; m != N_; ++m) {
            topic_count_[m] = topic_count_[m-1]+K_;
        }

        topic_area_ = new floating*[N_];
        topic_area_[0] = new floating[nr_areas_*K_];
        for (int a = 1; a != nr_areas_; ++a) {
            topic_area_[a] = topic_area_[a-1]+K_;
        }

        topic_term_ = new floating*[Nwords_];
        topic_term_[0] = new floating[Nwords_*K_];
        for (int t = 1; t != Nwords_; ++t) {
            topic_term_[t] = topic_term_[t-1] + K_;
        }

        topic_numeric_count_ = new floating*[F_];
        topic_numeric_count_[0] = new floating[F_*K_];
        for (int f = 1; f < F_; ++f) {
            topic_numeric_count_[f] = topic_numeric_count_[f-1] + K_;
        }
        sum_f_ = new floating[K_*F_];
        sum_f2_ = new floating[K_*F_];
    }

void lda::lda_collapsed::step() {
    int* z = zi_[0];
    floating zb_gamma[L_];
    for (int i = 0; i != N_; ++i) {
        floating z_bar[K_];
        const floating Ni = size(i);
        for (int k = 0; k != K_; ++k) {
            z_bar[k] = topic_count_[i][k] / floating(Ni);
        }
        for (int ell = 0; ell != L_; ++ell) {
            zb_gamma[ell] = dot_product(z_bar, gamma(ell), K_);
        }
        for (int k = 0; k != K_; ++k) {
            topic_[k] -= topic_count_[i][k];
        }

        for (const sparse_int* j = words_[i]; j->value != -1; ++j) {
            const int area = area_of(j->value);
            for (int cji = 0; cji != j->count; ++cji) {
                floating p[K_];
                const int ok = *z;
                --topic_count_[i][ok];
                --topic_area_[area][ok];
                --topic_term_[j->value][ok];
                floating psum = 0;
                for (int k = 0; k != K_; ++k) {
                    p[k] = ((topic_term_[j->value][k] + beta_) * (topic_count_[i][k] + alpha_)) /
                            ((topic_area_[area][k] + beta_) * (Ni + alpha_ - 1));
                    const floating* li = ls(i);
                    for (int ell = 0; ell != L_; ++ell) {
                        if (li[ell]) {
                            const floating delta = gamma(ell)[k] - gamma(ell)[ok];
                            p[k] *= phi(li[ell] * (zb_gamma[ell]+delta/Ni));
                        } else {
                            p[k] *= (.5);
                        }
                    }
                    psum += p[k];
                    assert(!std::isnan(p[k]));
                }
                p[K_-1] = psum + 1;
                floating rsum = psum*R.uniform01();
                int k = 0;
                while (rsum > p[k]) {
                    rsum -= p[k];
                    ++k;
                }

                *z++ = k;
                ++topic_count_[i][k];
                ++topic_area_[area][k];
                ++topic_term_[j->value][k];
                for (int ell = 0; ell != L_; ++ell) {
                    const floating* gl = gamma(ell);
                    zb_gamma[ell] -= gl[ok]/Ni;
                    zb_gamma[ell] += gl[k]/Ni;
                }
            }
        }
        for (int k = 0; k != K_; ++k) {
            topic_[k] += topic_count_[i][k];
        }
        for (int f = 0; f != F_; ++f) {
            const floating fv = features_[i][f];
            floating p[K_];
            const int ok = *z;
            --topic_count_[i][ok];
            --topic_numeric_count_[f][ok];
            floating* sf = sum_f(f);
            floating* sf2 = sum_f2(f);
            sf[ok] -= fv;
            sf2[ok] -= fv*fv;
            for (int k = 0; k != K_; ++k) {
                const floating n = topic_numeric_count_[f][k];
                const floating n_prime = Gn0_ + n;
                const floating f_bar = (n ? sf[k]/n : 0);
                const floating f2_bar = (n ? sf2[k]/n : 0.);
                const floating a_prime = Ga_ + n/2.;
                const floating b_prime = Gb_ + n/2.* (f2_bar- f_bar*f_bar) + .5*n*Gn0_*(f_bar - Gmu_)*(f_bar - Gmu_)/n_prime;

                const floating n_k = 1+n;
                const floating n_prime_k = n_prime + 1;
                const floating f_bar_k = (sf[k] + fv)/n_k;
                const floating f2_bar_k = (sf2[k] + fv*fv)/n_k;
                const floating a_prime_k = a_prime + 1./2;
                const floating b_prime_k = Gb_ + n_k/2.* (f2_bar_k- f_bar_k*f_bar_k) + .5*n_k*Gn0_*(f_bar_k - Gmu_)*(f_bar_k - Gmu_)/n_prime_k;
                assert(b_prime > 0);
                assert(a_prime > 0);
                p[k] = log(topic_count_[i][k] + alpha_);
                p[k] -= log(size(i) + alpha_ - 1);
                p[k] += 1./2.*log(n_prime/n_prime_k);
                p[k] += gsl_sf_lngamma(a_prime)   + a_prime   * log(b_prime);
                p[k] -= gsl_sf_lngamma(a_prime_k) + a_prime_k * log(b_prime_k);
                assert(!std::isnan(p[k]));
                for (int ell = 0; ell != L_; ++ell) {
                    const floating* li = ls(i);
                    if (li[ell]) {
                        const floating delta = gamma(ell)[k] - gamma(ell)[ok];
                        const floating e = phi(li[ell] * (zb_gamma[ell]+delta/Ni));
                        p[k] += std::log(e);
                    } else {
                        p[k] += -1; // std::log(.5);
                    }
                }
                assert(!std::isnan(p[k]));
            }

            const int k = categorical_sample_logps(R, p, K_);
            *z++ = k;
            ++topic_count_[i][k];
            ++topic_numeric_count_[f][k];
            sf[k] += fv;
            sf2[k] += fv*fv;
            for (int ell = 0; ell != L_; ++ell) {
                const floating* gl = gamma(ell);
                zb_gamma[ell] -= gl[ok]/Ni;
                zb_gamma[ell] += gl[k]/Ni;
            }
        }
    }
    this->update_gammas();
    //this->update_alpha_beta();
    this->verify();
}
void lda::lda_uncollapsed::step() {
    random_source R2 = R;
    R.uniform01();
    floating proposal[K_ * Nwords_];
    floating crossed[K_ * Nwords_];
    std::fill(proposal, proposal + K_*Nwords_, beta_);
    floating f_count[K_*F_];
    floating f_sum[K_*F_];
    floating f_sum2[K_*F_];
    std::fill(f_count, f_count + K_*F_, 0.);
    std::fill(f_sum, f_sum + K_*F_, 0.);
    std::fill(f_sum2, f_sum2 + K_*F_, 0.);
    const floating Vp_ = 8.;
    #pragma omp parallel firstprivate(R2) shared(proposal, crossed)
    {
        floating* priv_proposal = new floating[K_*Nwords_];
        floating* f_pcount = new floating[K_*F_];
        floating* f_priv = new floating[K_*F_];
        floating* f_priv2 = new floating[K_*F_];
        std::fill(priv_proposal, priv_proposal + K_*Nwords_, 0.);
        std::fill(f_pcount, f_pcount + K_*F_, 0.);
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

            for (const sparse_int* j = words_[i]; j->value != -1; ++j) {
                floating* crossed_j = crossed + (j->value)*K_;
                for (int cji = 0; cji != j->count; ++cji) {
                    const int cur = (ls_ ? zs_[i][wj] : 0);
                    assert (cur < K_);
                    for (int k = 0; k != K_; ++k) {
                        p[k] = crossed_j[k] * transfer[cur][k];
                    }
                    const int z  = categorical_sample(R2, p, K_);
                    if (ls_) zs_[i][wj++] = z;
                    assert(j->value < Nwords_);
                    assert(z < K_);
                    ++priv_proposal[z*Nwords_ + j->value];
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
                ++f_pcount[kf];
                f_priv[kf] += fif;
                f_priv2[kf] += fif*fif;
            }
            if (sample_[i]) {
                dirichlet_sample(R2, thetas(i), Tp, K_);
            }
            floating* zb = z_bar(i);
            std::copy(Tp, Tp + K_, zb);
            for (int k = 0; k != K_; ++k) {
                zb[k] /= Ni;
            }
            if (ls_) {
                floating* li = ls(i);
                floating* yi = ys(i);
                for (int ell = 0; ell != L_; ++ell) {
                    if (li[ell]) {
                        floating mu = dot_product(zb, gamma(ell), K_);
                        // Normalise:
                        const floating p2 = Vp_/(1.+Vp_);
                        mu *= li[ell] * p2;
                        floating s = left_truncated_normal(R2, -mu);
                        yi[ell] = mu + s/std::sqrt(p2);
                        yi[ell] *= li[ell];
                    }
                    //std::cerr << "mu_i: " << mu << "; ys_i: " << yi[ell] << " (ls_[i]: " << floating(ls_[i]) << ")\n";
                }
            }
        }

        #pragma omp critical
        {
            floating* p = priv_proposal;
            for (int kj = 0; kj < K_*Nwords_; ++kj) proposal[kj] += *p++;
            for (int kf = 0; kf < K_*F_; ++kf) {
                f_count[kf] += f_pcount[kf];
                f_sum[kf] += f_priv[kf];
                f_sum2[kf] += f_priv2[kf];
            }
        }
        delete [] priv_proposal;
        delete [] f_pcount;
        delete [] f_priv;
        delete [] f_priv2;

        #pragma omp barrier
        #pragma omp for
        for (int k = 0; k < K_; ++k) {
            logdirichlet_sample(R2, multinomials_[k], proposal+ k*Nwords_, Nwords_);
            for (int f = 0; f < F_; ++f) {
                // Check:
                // 'http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.126.4603&rep=rep1&type=pdf'
                // [page 8] or
                // http://en.wikipedia.org/wiki/Normal-gamma_distribution#Posterior_distribution_of_the_parameters
                const int kf = f*K_ + k;
                const floating n = f_count[kf];
                if (n) {
                    const floating sum = f_sum[kf];
                    const floating sum2 = f_sum2[kf];
                    const floating x_bar = sum/n;

                    const floating alpha_n  = Ga_ + n/2;
                    const floating beta_n = Gb_ + (sum2 - 2*sum*x_bar + n*x_bar*x_bar)/2. + n*Gn0_/2./(n+Gn0_)*(x_bar-Gmu_)*(x_bar-Gmu_);

                    const floating kappa_n  = Gn0_ + n;
                    const floating mu_n =  (Gn0_*Gmu_ + n * x_bar)/kappa_n;

                    normals_[k][f] = normal_gamma(R2, mu_n, kappa_n, alpha_n, beta_n);
                } else {
                    normals_[k][f] = normal_gamma(R2, Gmu_, Gn0_, Ga_, Gb_);
                }
            }
        }
        if (ls_) {
            // sample gamma
            double* const zdata = new double[N_ * K_];
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
            b.data = 0;
            b.block = 0;
            b.owner = 0;

            gsl_vector gammav;
            gammav.size = K_;
            gammav.stride = 1;
            gammav.block = 0;
            gammav.owner = 0;

            gsl_vector* r = gsl_vector_alloc(N_);
            gsl_vector* tau = gsl_vector_alloc(K_);

            #pragma omp for
            for (int ell = 0; ell < L_; ++ell) {
                // The operation below might actually clobber zdata
                std::copy(z_bars_, z_bars_ + N_ * K_, zdata);

                b.data = ys_ + ell;
                gammav.data = gamma(ell);

                gsl_linalg_QR_decomp(&Z, tau);
                gsl_linalg_QR_lssolve(&Z, tau, &b, &gammav, r);

            }
            gsl_vector_free(tau);
            gsl_vector_free(r);
            delete [] zdata;
        }
    }
}

void lda::lda_collapsed::verify() const {
    for (int i = 0; i != N_; ++i) {
        assert(
            std::accumulate(topic_count_[i], topic_count_[i] + K_, 0)
             == size(i));
    }
    for (int k = 0; k != K_; ++k) {
        int cdocs = 0;
        for (int i = 0; i != N_; ++i) cdocs += roundi(topic_count_[i][k]);

        int wc = 0;
        for (int j = 0; j != Nwords_; ++j) wc += roundi(topic_term_[j][k]);

        int fc = 0;
        for (int f = 0; f != F_; ++f) fc += roundi(topic_numeric_count_[f][k]);

        assert( cdocs == (wc + fc) );
    }
    for (int k = 0; k != K_; ++k) {
        int pa = 0;
        for (int a = 0; a != nr_areas_; ++a) pa += roundi(topic_area_[a][k]);
        assert( pa == topic_[k]);
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
            normals_[k][f] = normal_gamma(R, Gmu_, Gn0_, Ga_, Gb_);
        }
    }
    if (gamma_) std::fill(gamma_, gamma_ + K_ * L_, 0.);
}

void lda::lda_collapsed::forward() {
    std::fill(topic_, topic_ + K_, 0);
    std::fill(topic_count_[0], topic_count_[0] + N_*K_, 0);
    std::fill(topic_term_[0], topic_term_[0] + K_*Nwords_, 0);
    std::fill(topic_area_[0], topic_area_[0] + K_*nr_areas_, 0);

    std::fill(topic_numeric_count_[0], topic_numeric_count_[0] + F_*K_, 0);

    std::fill(sum_f_, sum_f_ + K_*F_, 0.);
    std::fill(sum_f2_, sum_f2_ + K_*F_, 0.);
    int* z = zi_[0];
    for (int i = 0; i != N_; ++i) {
        for (const sparse_int* j = words_[i]; j->value != -1; ++j) {
            const int area = area_of(j->value);
            for (int cji = 0; cji != j->count; ++cji) {
                const int k = R.random_int(0, K_);
                *z++ = k;
                ++topic_count_[i][k];
                ++topic_term_[j->value][k];
                ++topic_[k];
                ++topic_area_[area][k];
            }
        }
        for (int f = 0; f != F_; ++f) {
            floating* sf = sum_f(f);
            floating* sf2 = sum_f2(f);
            const floating fv = features_[i][f];
            const int k = R.random_int(0, K_);
            *z++ = k;
            ++topic_numeric_count_[f][k];
            ++topic_count_[i][k];
            ++topic_[k];
            sf[k] += fv;
            sf2[k] += fv*fv;
        }
    }
    std::fill(gamma_, gamma_ + K_ *L_, 0.);
}

void lda::lda_collapsed::update_alpha_beta() {
    floating ga = N_*K_ * (gsl_sf_psi  (alpha_) -    gsl_sf_psi  (K_*alpha_));
    floating H  = N_*K_ * (gsl_sf_psi_1(alpha_) - K_*gsl_sf_psi_1(K_*alpha_));
    for (int i = 0; i != N_; ++i) {
        for (int k = 0; k != K_; ++k) {
            const floating N_ij = topic_count_[i][k];
            ga += gsl_sf_psi  (N_ij + alpha_);
            H  += gsl_sf_psi_1(N_ij + alpha_);
        }
        ga -=    K_ * gsl_sf_psi  (K_*alpha_ + floating(size(i)));
        H  -= K_*K_ * gsl_sf_psi_1(K_*alpha_ + floating(size(i)));
    }
    alpha_ -= ga/H;
}


void lda::lda_collapsed::update_gammas() {
    // This is performing a Newton-Raphson step
    // We perform a single step per iteration
    if (!L_) return;
    using boost::scoped_array;
    scoped_array<double> gdata(new double[L_*K_]);
    scoped_array<double> Hdata(new double[L_*K_*K_]);

    std::fill(gdata.get(), gdata.get() + L_*K_, 0.);
    std::fill(Hdata.get(), Hdata.get() + L_*K_*K_, 0.);

    const floating sq_tpi = 0.3989422804014327; // sqrt(1/2./pi)
    const bool using_l2_penalty = false;
    const floating sigma2_over_1_plus_sigma2 = 8./9;
    for (int i = 0; i != N_; ++i) {
        const floating* li = ls(i);
        floating zi[K_];
        for (int k = 0; k != K_; ++k) {
            // The factor sigma2_over_1_plus_sigma2 is used below, but the code
            // is somewhat simpler if we insert it here
            zi[k] = sigma2_over_1_plus_sigma2*(topic_count_[i][k] + alpha_)/(size(i) + K_*alpha_);
        }
        for (int ell = 0; ell < L_; ++ell) {
            if (li[ell]) {
                const floating f_g = li[ell]*dot_product(gamma(ell), zi, K_);
                const floating phi0 = phi(f_g);
                const floating phi1 = sq_tpi*std::exp(-f_g*f_g/2.);
                const floating phi2 = -f_g*phi1;
                const floating gfactor = phi1/phi0;
                const floating hfactor = (phi2/phi0 - gfactor*gfactor);

                double* g = gdata.get() + K_*ell;
                double* H = Hdata.get() + K_*K_*ell;

                for (int k = 0; k != K_; ++k) {
                    g[k] += li[ell] * zi[k] * gfactor;
                    for (int m = 0; m != K_; ++m) {
                        H[k*K_+m] += zi[k]*zi[m]*hfactor;
                    }
                }
            }
        }
    }
    scoped_array<double> ddata(new double[K_]);
    gsl_vector_view dvector = gsl_vector_view_array(ddata.get(), K_);
    gsl_permutation* permutation = gsl_permutation_alloc(K_);
    for (int ell = 0; ell != L_; ++ell) {
        floating* gl = gamma(ell);
        double* g = gdata.get() + K_*ell;
        double* H = Hdata.get() + ell*K_*K_;
        for (int k = 0; k != K_; ++k) {
            if (using_l2_penalty) {
                g[k] += -2*lambda_*gl[k];
                H[k*K_+k] += -2*lambda_;
            } else {
                if (gl[k] < 0) g[k] += lambda_;
                else if (gl[k] > 0) g[k] -= lambda_;
            }
        }

        int signum;
        gsl_vector_view gvector = gsl_vector_view_array(gdata.get() + ell*K_, K_);
        gsl_matrix_view Hmatrix = gsl_matrix_view_array(H, K_, K_);
        gsl_linalg_LU_decomp(&Hmatrix.matrix, permutation, &signum);
        if (has_diagonal_zero(&Hmatrix.matrix, K_)) continue;
        gsl_linalg_LU_solve(&Hmatrix.matrix, permutation, &gvector.vector, &dvector.vector);


        for (int k = 0; k != K_; ++k) {
            const double nval = gl[k] - ddata[k];
            if (using_l2_penalty) {
                gl[k] = nval;
            } else {
                if ((nval * gl[k]) >= 0) gl[k] = nval;
                else gl[k] = 0;
            }
        }
    }
    gsl_permutation_free(permutation);
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
            // compute p += \sum_j w_j  * log( \sum_k \theta_k \psi_k )
            // we use an intermediate variable (local_p) to avoid adding really
            // small numbers to a larger number
            floating local_p = 0;
            for (const sparse_int* j = words_[i]; j->value != -1; ++j) {
                floating sum_k = 0.;
                floating* crossed_j = crossed + (j->value)*K_;
                for (int k = 0; k != K_; ++k) {
                    sum_k += Ti[k] * crossed_j[k];
                }
                local_p += j->value * (std::log(sum_k) + offset[j->value]);
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



floating lda::lda_collapsed::logperplexity(const std::vector<int>& words, const std::vector<float>& fs, const std::vector<float>& labels) const {
    if (int(fs.size()) != F_) return -1;
    if (!(labels.empty() || int(labels.size()) == L_)) return -1;
    if (words.size() && (
        (*std::max_element(words.begin(), words.end()) >= Nwords_) ||
        (*std::min_element(words.begin(), words.end()) < 0)
        )) return -2;
    std::vector<int> zs;
    floating counts[K_];
    floating thetas[K_];
    this->sample_one(words, fs, zs, counts);
    for (int k = 0; k != K_; ++k) {
        thetas[k] = counts[k] + alpha_;
        thetas[k] /= (words.size()+K_*alpha_);
    }
    floating logp = dirichlet_logP_uniform(thetas, alpha_, K_, true);

    for (int k = 0; k != K_; ++k) {
        logp += counts[k] * log(thetas[k]);
    }

    for (unsigned j = 0; j != words.size(); ++j) {
        const int w = words[j];
        const int k = zs[j];
        const int a = area_of(w);
        logp += log(floating(topic_term_[w][k] + beta_)/(topic_area_[a][k] + area_size(a)*beta_));
    }
    for (int f = 0; f != F_; ++f) {
        throw "This is not implemented.";
    }
    if (!labels.empty()) {
        for (int ell = 0; ell != L_; ++ell) {
            const floating* gl = gamma(ell);
            const floating val = dot_product(gl, thetas, K_);
            const floating p = truncated_normal_like(val, labels[ell]);
            //std::cerr << "val: " << val << "\n\tp: " << p << "\tlog(p): " << std::log(p) << '\n';
            logp += std::log(p);
        }
    }
    return logp;
}


floating lda::lda_collapsed::logP(bool normalise) const {
    floating logp = 0;
    if (normalise) {
        throw "not implemented.";
        logp += N_ * ( gsl_sf_lngamma(K_ * alpha_) - K_* gsl_sf_lngamma(alpha_));
    }
    for (int i = 0; i < N_; ++i) {
        for (int k = 0; k != K_; ++k) {
            logp += gsl_sf_lngamma(topic_count_[i][k] + alpha_);
        }
    }
    for (int j = 0; j < Nwords_; ++j) {
        for (int k = 0; k != K_; ++k) {
            logp += gsl_sf_lngamma(topic_term_[j][k] + beta_);
        }
    }
    for (int f = 0; f < F_; ++f) {
        const floating* sf = sum_f(f);
        const floating* sf2 = sum_f2(f);

        for (int k = 0; k != K_; ++k) {
            const floating n = topic_numeric_count_[f][k];
            const floating n_prime = Gn0_ + n;
            const floating f_bar = (n ? sf[k]/n : 0);
            const floating f2_bar = (n ? sf2[k]/n : 0.);
            const floating a_prime = Ga_ + n/2.;
            const floating b_prime = Gb_ + n/2.* (f2_bar- f_bar*f_bar) + .5*n*Gn0_*(f_bar - Gmu_)*(f_bar - Gmu_)/n_prime;
            logp += gsl_sf_lngamma(a_prime)+a_prime * log(b_prime);
            logp += 1./2.*log(n_prime);
        }
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
        const int* z = zi_[0];
        for (int i = 0; i != N_; ++i) {
            for (const sparse_int* j = words_[i]; j->value != -1; ++j) {
                for (int cji = 0; cji != j->count; ++cji) {
                    if (*z++ == k) ++multinomials[j->value];
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
    const int* z = zi_[0];
    for (int i = 0; i != N_; ++i) {
        std::fill(thetas, thetas + K_, alpha_);
        for (const sparse_int* j = words_[i]; j->count != -1; ++j) {
            for (int cji = 0; cji != j->count; ++cji) {
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
void lda::lda_uncollapsed::save_model(std::ostream& out) const {
    using std::endl;
    out << N_ << endl;
    out << K_ << endl;
    out << F_ << endl;
    out << L_ << endl;
    out << Nwords_ << endl;
    out << alpha_ << endl;
    out << beta_ << endl;
    out << lambda_ << endl;
    out << Ga_ << endl;
    out << Gb_ << endl;
    out << Gn0_ << endl;
    out << Gmu_ << endl;
    out << endl;
    out << endl;
    for (int k = 0; k != K_; ++k) {
        const floating* m = multinomials_[k];
        for (int j = 0; j != Nwords_; ++j) {
            out << m[j] << '\t';
        }
        out << endl;

        for (int f = 0; f != F_; ++f) {
            out << normals_[k][f].mu << ' ' << normals_[k][f].precision << '\t';
        }
        out << endl;
    }
    out << endl;
    for (int i = 0; i != N_; ++i) {
        for (const sparse_int* j = words_[i]; j->count != -1; ++j) {
            for (int cji = 0; cji != j->count; ++cji) {
                out << j->value << '\t';
            }
        }
        out << endl;

        const floating* T = thetas(i);
        for (int k = 0; k != K_; ++k) {
            out << T[k] << '\t';
        }
        out << endl;

        for (int f = 0; f != F_; ++f) {
            out << features_[i][f] << '\t';
        }
        out << endl;

        if (ls_) {
            const int* zs = zs_[i];
            const int Ni = zs[0];
            for (int j = 0; j != (1+Ni); ++j) {
                out << zs[j] << '\t';
            }
            out << endl;

            const floating* li = ls(i);
            const floating* yi = ys(i);
            for (int ell = 0; ell != L_; ++ell) {
                out << yi[ell] << ' ' << li[ell] << '\t';
            }
            out << endl;
        }
    }
    if (ls_) {
        for (int ell = 0; ell != L_; ++ell) {
            const floating* gl = gamma(ell);
            for (int k = 0; k != K_; ++k) {
                out << gl[k] << '\t';
            }
            out << endl;
        }
    }
}
void lda::lda_uncollapsed::load_model(std::istream& in) {
#define CHECK(type,var) do { type t; in >> t; if (var != t) { throw "Check failed: " #var " is not what is expected\n"; } } while(0)
    CHECK(int, N_);
    CHECK(int, K_);
    CHECK(int, F_);
    CHECK(int, L_);
    CHECK(int, Nwords_);
    CHECK(floating, alpha_);
    CHECK(floating, beta_);
    CHECK(floating, lambda_);
    CHECK(floating, Ga_);
    CHECK(floating, Gb_);
    CHECK(floating, Gn0_);
    CHECK(floating, Gmu_);

    for (int k = 0; k != K_; ++k) {
        floating* m = multinomials_[k];
        for (int j = 0; j != Nwords_; ++j) {
            in >> m[j];
        }

        for (int f = 0; f != F_; ++f) {
            in >> normals_[k][f].mu >> normals_[k][f].precision;
        }
    }

    for (int i = 0; i != N_; ++i) {
        for (const sparse_int* j = words_[i]; j->count != -1; ++j) {
            for (int cji = 0; cji != j->count; ++cji) {
                int t;
                in >> t;
                if (t != j->value) {
                    throw "j->value was expected\n";
                }
            }
        }

        floating* T = thetas(i);
        for (int k = 0; k != K_; ++k) {
            in >> T[k];
        }

        for (int f = 0; f != F_; ++f) {
            floating t;
            in >> t;
            /*if (t != features_[i][f]) {
                throw "features_[i][f] was expected\n";
            }*/
        }

        if (ls_) {
            int* zs = zs_[i];
            const int Ni = zs[0];
            for (int j = 0; j != (1+Ni); ++j) {
                in >> zs[j];
            }

            floating* yi = ys(i);
            floating t;
            for (int ell = 0; ell != L_; ++ell) {
                in >> yi[ell] >> t;
            }
        }
    }
    if (ls_) {
        for (int ell = 0; ell != L_; ++ell) {
            floating* gl = gamma(ell);
            for (int k = 0; k != K_; ++k) {
                in >> gl[k];
            }
        }
    }
}
void lda::lda_collapsed::save_model(std::ostream& out) const {
    using std::endl;
    out << N_ << endl;
    out << K_ << endl;
    out << F_ << endl;
    out << L_ << endl;
    out << Nwords_ << endl;
    out << alpha_ << endl;
    out << beta_ << endl;
    out << lambda_ << endl;
    out << Ga_ << endl;
    out << Gb_ << endl;
    out << Gn0_ << endl;
    out << Gmu_ << endl;
    out << endl;
    out << endl;
    out << area_markers_.size();
    for (int a = 0; a != area_markers_.size(); ++a) {
        out << ' ' << area_markers_[a];
    }
    out << endl;
    out << endl;
    out << endl;
    for (int i = 0; i != N_; ++i) {
        const int Ni = size(i);
        out << Ni << ' ';
        const int* z = zi_[i];
        for (int zi = 0; zi != Ni; ++zi, ++z) {
            out << *z << ' ';
        }
        out << endl;
    }
    for (int ell = 0; ell != L_; ++ell) {
        const floating* gl = gamma(ell);
        for (int k = 0; k != K_; ++k) {
            out << gl[k];
        }
        out << endl;
    }
}

template<typename T>
void check_value(std::istream& in, const T val, const char* what = "") {
    T t;
    in >> t;
    if (val != t) {
        std::ostringstream out;
        out << "Loading check for '" << what << "' failed (expected " << val << ", got " << t << ").";
        throw out.str();
    }
}

void lda::lda_collapsed::load_model(std::istream& in) {
    check_value<int>(in, N_, "N");
    check_value<int>(in, K_, "K");
    check_value<int>(in, F_, "F");
    check_value<int>(in, L_, "L");
    check_value<int>(in, Nwords_, "Nwords");

    check_value<floating>(in, alpha_, "a");
    check_value<floating>(in, beta_, "b");
    check_value<floating>(in, lambda_, "lambda");
    check_value<floating>(in, Ga_, "Ga");
    check_value<floating>(in, Gb_, "Gb");
    check_value<floating>(in, Gn0_, "Gno");
    check_value<floating>(in, Gmu_, "Gmu");

    int areas;
    in >> areas;
    if (areas <= 0) throw "Need at least one area!";
    area_markers_.resize(areas);
    for (int a = 0; a != areas; ++a) in >> area_markers_[a];


    std::fill(topic_, topic_ + K_, 0);
    std::fill(topic_count_[0], topic_count_[0] + N_*K_, 0);
    std::fill(topic_term_[0], topic_term_[0] + K_*Nwords_, 0);

    std::fill(topic_numeric_count_[0], topic_numeric_count_[0] + F_*K_, 0);

    std::fill(sum_f_, sum_f_ + K_*F_, 0.);
    std::fill(sum_f2_, sum_f2_ + K_*F_, 0.);

    for (int i = 0; i != N_; ++i) {
        const int Ni = size(i);
        check_value<int>(in, Ni);
        int* z = zi_[i];
        for (const sparse_int* j = words_[i]; j->value != -1; ++j) {
            for (int cj = 0; cj != j->count; ++cj) {
                int k;
                in >> k;
                *z++ = k;
                ++topic_[k];
                ++topic_count_[i][k];
                ++topic_term_[j->value][k];
            }
        }
        for (int f = 0; f != F_; ++f) {
            const floating fv = features_[i][f];
            int k;
            in >> k;
            *z++ = k;
            ++topic_[k];
            ++topic_numeric_count_[f][k];
            sum_f(f)[k] += fv;
            sum_f2(f)[k] += fv*fv;
        }
    }
    for (int ell = 0; ell != L_; ++ell) {
        floating* gl = gamma(ell);
        for (int k = 0; k != K_; ++k) {
            in >> gl[k];
        }
    }

}


int lda::lda_uncollapsed::retrieve_logbeta(int k, float* res, int size) const {
    if (size != Nwords_) return 0;
    std::copy(multinomials_[k], multinomials_[k] + Nwords_, res);
    return Nwords_;
}
int lda::lda_uncollapsed::retrieve_theta(int i, float* res, int size) const {
    if (i >= N_) return -1;
    if (size != K_) return 0;
    const floating* m = thetas_ + i*K_;
    std::copy(m, m + K_, res);
    return K_;
}
int lda::lda_collapsed::retrieve_theta(int i, float* res, int s) const {
    if (i >= N_) return -1;
    if (s != K_) return 0;
    std::fill(res, res + K_, 0);
    for (const int* z = zi_[i]; z != zi_[i+1]; ++z) {
        ++res[*z];
    }
    for (int k = 0; k != K_; ++k) {
        res[k] /= size(i);
    }
    return K_;
}
int lda::lda_base::retrieve_gamma(int ell, float* res, int size) const {
    if (size != K_) return 0;
    if (ell > L_) return -1;
    std::copy(gamma(ell), gamma(ell + 1), res);
    return K_;
}
int lda::lda_uncollapsed::retrieve_ys(int i, float* res, int size) const {
    if (i >= N_) return -1;
    if (size != L_) return 0;
    std::copy(ys(i), ys(i+1), res);
    return L_;
}

int lda::lda_uncollapsed::retrieve_z_bar(int i, float* res, int size) const {
    if (i >= N_) return -1;
    if (size != K_) return 0;
    std::copy(z_bar(i), z_bar(i+1), res);
    return K_;
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

int lda::lda_uncollapsed::project_one(const std::vector<int>& words, const std::vector<float>& fs, float* res, int size) const {
    if (size != K_) return 0;
    floating thetas[K_];
    this->sample_one(words, fs, thetas);
    std::copy(thetas, thetas + K_, res);
    return size;
}

int lda::lda_collapsed::project_one(const std::vector<int>& words, const std::vector<float>& fs, float* res, int size) const {
    if (size != K_) return 0;
    if (int(fs.size()) != F_) return -1;
    if ((*std::max_element(words.begin(), words.end()) >= Nwords_) ||
        (*std::min_element(words.begin(), words.end()) < 0)) return -2;
    std::vector<int> zs;
    floating counts[K_];
    this->sample_one(words, fs, zs, counts);
    for (int k = 0; k != K_; ++k) res[k] = floating(counts[k])/zs.size();
    assert(std::accumulate(counts, counts + K_, 0.) == floating(zs.size()));
    return size;
}

void lda::lda_collapsed::sample_one(const std::vector<int>& words, const std::vector<float>& fs, std::vector<int>& zs, floating counts[]) const {
    zs.reserve(words.size() + fs.size());
    std::fill(counts, counts + K_, 0);

    random_source R2 = R;
    for (unsigned zi = 0; zi != (words.size() + fs.size()); ++zi) {
        const int z = R2.random_int(0,K_);
        zs.push_back(z);
        ++counts[z];
    }
    for (int it = 0; it != 32; ++it) {
        for (unsigned zi = 0; zi != zs.size(); ++zi) {
            const int ok = zs[zi];
            --counts[ok];

            floating p[K_];
            int (*sample_function)(random_source&, const floating*, int);
            if (zi < words.size()) {
                sample_function = categorical_sample_cps;
                for (int k = 0; k != K_; ++k) {
                    const int w = words[zi];
                    p[k] = (topic_term_[w][k] + beta_)/
                                (topic_[k] + beta_) *
                        (counts[k] + alpha_)/(zs.size() + alpha_ - 1);
                    if (k > 0) p[k] += p[k-1];
                }
                for (int k = 0; k != K_; ++k) p[k] /= p[K_-1];
            } else {
                sample_function = categorical_sample_logps;
                for (int k = 0; k != K_; ++k) {
                    const int f = zi-words.size();
                    const floating fv = fs[f];
                    const floating* sf = sum_f(f);
                    const floating* sf2 = sum_f2(f);

                    const floating n = topic_numeric_count_[f][k];
                    const floating n_prime = Gn0_ + n;
                    const floating f_bar = (n ? sf[k]/n : 0);
                    const floating f2_bar = (n ? sf2[k]/n : 0.);
                    const floating a_prime = Ga_ + n/2.;
                    const floating b_prime = Gb_ + n/2.* (f2_bar- f_bar*f_bar) + .5*n*Gn0_*(f_bar - Gmu_)*(f_bar - Gmu_)/n_prime;

                    const floating n_k = 1+n;
                    const floating n_prime_k = n_prime + 1;
                    const floating f_bar_k = (sf[k] + fv)/n_k;
                    const floating f2_bar_k = (sf2[k] + fv*fv)/n_k;
                    const floating a_prime_k = Ga_ + n_k/2.;
                    const floating b_prime_k = Gb_ + n_k/2.* (f2_bar_k- f_bar_k*f_bar_k) + .5*n_k*Gn0_*(f_bar_k - Gmu_)*(f_bar_k - Gmu_)/n_prime_k;
                    assert(b_prime > 0);
                    assert(a_prime > 0);
                    p[k] = log(counts[k] + alpha_)+gsl_sf_lngamma(a_prime)+a_prime * log(b_prime);
                    p[k] += 1./2.*log(n_prime/n_prime_k);
                    p[k] -= gsl_sf_lngamma(a_prime_k)+a_prime_k * log(b_prime_k);
                }
            }
            const int k = sample_function(R2, p, K_);
            assert(k < K_);
            zs[zi] = k;
            ++counts[k];
        }
    }
}

float lda::lda_base::score_one(int ell, const float* res, int size) const {
    if (ell >= L_) return -1;
    if (size != K_) return 0;
    return dot_product(res, gamma(ell), size);
}

floating lda::lda_uncollapsed::logperplexity(const std::vector<int>& words, const std::vector<float>& fs, const std::vector<float>& labels) const {
    floating thetas[K_];
    this->sample_one(words, fs, thetas);
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
    for (unsigned int j = 0; j != words.size(); ++j) {
        floating sum_k = 0;
        floating* crossed_j = crossed + words[j]*K_;
        for (int k = 0; k != K_; ++k) {
            sum_k += thetas[k] * crossed_j[k];
        }
        logp += std::log(sum_k) + offset[j];
    }

    if (!labels.empty()) {
        throw "labels not supported";
    }
    return logp;
}

void lda::lda_uncollapsed::sample_one(const std::vector<int>& words, const std::vector<float>& fs, floating* thetas) const {
    random_source R2 = R;
    const int nr_iters = 20;
    std::fill(thetas, thetas + K_, 1.);

    floating crossed[K_ * Nwords_];
    floating normals[F_][K_];
    const int docsize = words.size();

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
    for (int f = 0; f < F_; ++f) {
        for (int k = 0; k != K_; ++k) {
            normals[f][k] = normal_like(fs[f], normals_[k][f]);
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
            int z = categorical_sample(R2, p, K_);
            ++proposal[z];
        }
        for (int f = 0; f != F_; ++f) {
            floating p[K_];
            for (int k = 0; k != K_; ++k) {
                p[k] = thetas[k] * normals[f][k];
            }
            int z = categorical_sample(R2, p, K_);
            ++proposal[z];
        }
        dirichlet_sample(R2, thetas, proposal, K_);
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



