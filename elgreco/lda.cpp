#include <cmath>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <limits>
#include <assert.h>
#include <boost/scoped_array.hpp>
#include <boost/scoped_ptr.hpp>

#include <omp.h>

#include "lda.h"

#include <gsl/gsl_linalg.h>
#include <gsl/gsl_sf.h>

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

floating phi(const double x) {
    /* The table was generated from the following programme:
     *
     * from scipy import special
     * def phi(x):
     *     return (1.+special.erf(x))/2.
     *     data = [phi(i/100.) for i in xrange(200)]
     *     print data
     */
    assert(!std::isnan(x));
    if (x < 0) return 1.-phi(-x);
    if (x > 2) return .99;
    const floating table[] = {0.5, 0.50564170777792483, 0.51128228734592251, 0.51692061117086774, 0.52255555307256241, 0.52818598889850832, 0.53381079719665425, 0.53942885988544542, 0.54503906292050908, 0.55064029695731342, 0.55623145800914242, 0.56181144809973715, 0.56737917590996001, 0.57293355741784791, 0.57847351653142787, 0.58399798571368178, 0.58950590659905289, 0.59499623060090445, 0.60046791950934786, 0.60591994607887489, 0.61135129460523918, 0.61676096149105175, 0.62214795579956439, 0.62751129979613651, 0.63285002947689595, 0.63816319508411845, 0.64344986160787454, 0.64870910927350645, 0.65394003401451695, 0.65914174793047609, 0.66431337972956372, 0.66945407515539512, 0.67456299739779135, 0.67963932748717948, 0.68468226467232929, 0.68969102678115513, 0.69466485056433203, 0.69960299202149956, 0.70450472670984698, 0.70936935003489809, 0.7141961775233342, 0.71898454507771969, 0.72373380921301267, 0.72844334727477011, 0.73311255763897876, 0.73774085989346183, 0.74232769500083984, 0.74687252544304106, 0.75137483534738236, 0.75583413059426152, 0.76024993890652326, 0.76462180992058526, 0.76894931523942722, 0.77323204846757077, 0.77746962522819518, 0.78166168316255447, 0.78580788191188422, 0.78990790308199799, 0.79396145019080033, 0.79796824859895432, 0.80192804542396301, 0.80584060943794011, 0.8097057309493606, 0.81352322166909785, 0.81729291456107056, 0.82101466367783593, 0.82468834398147717, 0.82831385115015244, 0.8318911013706789, 0.83542003111753882, 0.83890059691870922, 0.84233277510872218, 0.84571656156937558, 0.84905197145852229, 0.85233903892737284, 0.8555778168267576, 0.85876837640279535, 0.86191080698242972, 0.86500521564928934, 0.8680517269103456, 0.8710504823538302, 0.87400164029889482, 0.87690537543748115, 0.87976187846888643, 0.88257135572749723, 0.88533402880417622, 0.88805013416177836, 0.89071992274527534, 0.89334365958696627, 0.89592162340724757, 0.89845410621141597, 0.90094141288297058, 0.90338386077388089, 0.90578177929227888, 0.90813550948803123, 0.91044540363663895, 0.91271182482190905, 0.91493514651783348, 0.91711575217010388, 0.91925403477768486, 0.9213503964748575, 0.92340524811413838, 0.92541900885047101, 0.9273921057270742, 0.92932497326332575, 0.93121805304504834, 0.93307179331755408, 0.93488664858179327, 0.93666307919394476, 0.93840155096876909, 0.94010253478704087, 0.94176650620735902, 0.94339394508262742, 0.9449853351814812, 0.94654116381492837, 0.94806192146845747, 0.94954810143985602, 0.95100019948296777, 0.95241871345760842, 0.95380414298584248, 0.95515698911481772, 0.95647775398633472, 0.95776694051332334, 0.95902505206338073, 0.96025259214951486, 0.96145006412822909, 0.96261797090506473, 0.96375681464771235, 0.96486709650678915, 0.96594931634436687, 0.96700397247032621, 0.96803156138659974, 0.96903257753935568, 0.97000751307916511, 0.97095685762918271, 0.97188109806136203, 0.97278071828071655, 0.97365619901762601, 0.97450801762818129, 0.97533664790254826, 0.97614255988132448, 0.97692621967985271, 0.97768808932044804, 0.97842862657248442, 0.97914828480028238, 0.97984751281872962, 0.98052675475655904, 0.98118644992720294, 0.98182703270713445, 0.98244893242160214, 0.98305257323765538, 0.98363837406435584, 0.98420674846006162, 0.98475810454666779, 0.98529284493068192, 0.98581136663100621, 0.9863140610133001, 0.98680131373078361, 0.98727350467134845, 0.9877310079108339, 0.98817419167232201, 0.98860341829130927, 0.9890190441866018, 0.989421419836785, 0.98981088976211606, 0.99018779251168021, 0.99055246065566105, 0.99090522078256327, 0.99124639350123234, 0.99157629344751308, 0.99189522929538732, 0.99220350377243416, 0.99250141367945288, 0.99278924991409023, 0.99306729749831646, 0.99333583560959116, 0.99359513761556506, 0.99384547111216115, 0.99408709796488415, 0.99432027435320403, 0.99454525081786538, 0.99476227231097214, 0.99497157824870386, 0.99517340256651532, 0.99536797377668129, 0.99555551502804285, 0.99573624416781981, 0.99591037380535341, 0.9960781113776469, 0.99623965921657398, 0.9963952146176287, 0.99654496991009167, 0.99668911252849246, 0.99682782508524825, 0.99696128544436624, 0.99708966679609456, 0.99721313773241393, 0.99733186232326498, 0.99744600019340679, 0.99755570659980852};

    const int idx = int(x * 100);
    if (idx > 100) return .99;
    return table[idx];
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
        if (alpha_ <= 0) {
            throw "elgreco.lda: alpha must be strictly positive";
        }
        if (beta_ <= 0) {
            throw "elgreco.lda: beta must be strictly positive";
        }
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
            int ci = 0;
            while (ci < input.size(i)) {
                const int anchor = input(i, ci++);
                int c = 1;
                while (ci < input.size(i) && input(i, ci) == anchor) {
                    ++ci;
                    ++c;
                }
                *j++ = anchor;
                *cj++ = c;
            }
            *j++ = -1;
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
        z_ = new int[words.nr_words() + N_*F_];
        zi_ = new int*[N_+1];
        size_ = new int[N_];
        zi_[0] = z_;
        int* zinext = z_;
        for (int i = 0; i != N_; ++i) {
            int c = 0;
            for (const int* j = counts_idx_[i], *cj = counts_[i]; *j != -1; ++j, ++cj) {
                for (int cij = 0; cij != *cj; ++cij) {
                    ++zinext;
                    ++c;
                }
            }
            size_[i] = (c + F_);
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

        topic_numeric_count_ = new int*[F_];
        topic_numeric_count_[0] = new int[F_*K_];
        for (int f = 1; f < F_; ++f) {
            topic_numeric_count_[f] = topic_numeric_count_[f-1] + K_;
        }
        sum_f_ = new floating[K_*F_];
        sum_f2_ = new floating[K_*F_];
    }

void lda::lda_collapsed::step() {
    int* z = z_;
    floating* zb_gamma = new floating[L_];
    for (int i = 0; i != N_; ++i) {
        floating z_bar[K_];
        std::fill(z_bar, z_bar + K_, 0);
        const int Ni = size_[i];
        for (int d = 0; d != Ni; ++d) {
            ++z_bar[z[d]];
        }
        for (int k = 0; k != K_; ++k) z_bar[k] /= Ni;
        for (int ell = 0; ell != L_; ++ell) {
            zb_gamma[ell] = dot_product(z_bar, gamma(ell), K_);
        }

        for (const int* j = counts_idx_[i], *cj = counts_[i]; *j != -1; ++j, ++cj) {
            for (int cij = 0; cij != *cj; ++cij) {
                floating p[K_];
                const int ok = *z;
                --topic_count_[i][ok];
                --topic_[ok];
                --topic_term_[*j][ok];
                for (int k = 0; k != K_; ++k) {
                    p[k] = (topic_term_[*j][k] + beta_)/
                                (topic_[k] + beta_) *
                        (topic_count_[i][k] + alpha_)/
                                (size_[i] + alpha_ - 1);
                    for (int ell = 0; ell != L_; ++ell) {
                        const floating delta = gamma(ell)[k] - gamma(ell)[ok];
                        const floating s = ls(i)[ell] ? +1 : -1;
                        p[k] *= phi(s * (zb_gamma[ell]+delta/Ni));
                    }
                    assert(!std::isnan(p[k]));
                    if (k > 0) p[k] += p[k-1];
                }
                for (int k = 0; k != K_; ++k) p[k] /= p[K_-1];
                const int k = categorical_sample_cps(R, p, K_);

                *z++ = k;
                ++topic_count_[i][k];
                ++topic_[k];
                ++topic_term_[*j][k];
                for (int ell = 0; ell != L_; ++ell) {
                    const floating* gl = gamma(ell);
                    zb_gamma[ell] -= gl[ok]/Ni;
                    zb_gamma[ell] += gl[k]/Ni;
                }
            }
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
                const floating a_prime_k = Ga_ + n_k/2.;
                const floating b_prime_k = Gb_ + n_k/2.* (f2_bar_k- f_bar_k*f_bar_k) + .5*n_k*Gn0_*(f_bar_k - Gmu_)*(f_bar_k - Gmu_)/n_prime_k;
                assert(b_prime > 0);
                assert(a_prime > 0);
                p[k] = log(topic_count_[i][k] + alpha_);
                p[k] += 1./2.*log(n_prime/n_prime_k);
                p[k] += gsl_sf_lngamma(a_prime)   + a_prime   * log(b_prime);
                p[k] -= gsl_sf_lngamma(a_prime_k) + a_prime_k * log(b_prime_k);
                assert(!std::isnan(p[k]));
                for (int ell = 0; ell != L_; ++ell) {
                    const floating delta = gamma(ell)[k] - gamma(ell)[ok];
                    const floating s = ls(i)[ell] ? +1 : -1;
                    const floating e = phi(s * (zb_gamma[ell]+delta/Ni));
                    p[k] += std::log(e);
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
    for (int g = 0; g != 4; ++g) {
        this->solve_gammas();
    }
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
                    floating mu = dot_product(zb, gamma(ell), K_);
                    // Normalise:
                    const floating p2 = Vp_/(1.+Vp_);
                    mu *= p2;
                    if (!li[ell]) mu = -mu;
                    floating s = left_truncated_normal(R2, -mu);
                    yi[ell] = mu + s/std::sqrt(p2);
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
             == size_[i]);
    }
    for (int k = 0; k != K_; ++k) {
        int cdocs = 0;
        for (int i = 0; i != N_; ++i) cdocs += topic_count_[i][k];

        int wc = 0;
        for (int j = 0; j != Nwords_; ++j) wc += topic_term_[j][k];

        int fc = 0;
        for (int f = 0; f != F_; ++f) fc += topic_numeric_count_[f][k];

        assert( cdocs == (wc + fc) );
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

    std::fill(topic_numeric_count_[0], topic_numeric_count_[0] + F_*K_, 0);

    std::fill(sum_f_, sum_f_ + K_*F_, 0.);
    std::fill(sum_f2_, sum_f2_ + K_*F_, 0.);
    int* z = z_;
    for (int i = 0; i != N_; ++i) {
        for (const int* j = counts_idx_[i], *cj = counts_[i]; *j != -1; ++j, ++cj) {
            for (int cji = 0; cji != (*cj); ++cji) {
                const int k = R.random_int(0, K_);
                *z++ = k;
                ++topic_count_[i][k];
                ++topic_term_[*j][k];
                ++topic_[k];
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
    this->solve_gammas();
}

void lda::lda_collapsed::solve_gammas() {
    if (!L_) return;
    using boost::scoped_array;
    scoped_array<double> zbars(new double[N_ * K_]);
    scoped_array<double> y(new double[N_]);
    gsl_matrix Z;
    Z.size1 = N_;
    Z.size2 = K_;
    Z.tda = K_;
    Z.data = zbars.get();
    Z.block = 0;
    Z.owner = 0;

    gsl_vector b;
    b.size = N_;
    b.stride = 1;
    b.data = y.get();
    b.block = 0;
    b.owner = 0;

    gsl_vector gammav;
    gammav.size = K_;
    gammav.stride = 1;
    gammav.block = 0;
    gammav.owner = 0;

    gsl_vector* r = gsl_vector_alloc(N_);
    gsl_vector* tau = gsl_vector_alloc(K_);

    for (int i = 0; i != N_; ++i) {
        floating* zb = zbars.get() + K_*i;
        for (int k = 0; k != K_; ++k) {
            zb[k] = (topic_count_[i][k] + alpha_)/(size_[i] + K_*alpha_);
        }
    }

    gsl_linalg_QR_decomp(&Z, tau);
    for (int ell = 0; ell < L_; ++ell) {
        floating* gl = gamma(ell);
        for (int i = 0; i != N_; ++i) {
            const floating* li = ls(i);
            const floating p2 = 8./9.;
            const floating p2i = 9./8.;

            floating mu = dot_product(gl, zbars.get() + (K_*i), K_);
            mu *= p2;
            if (!li[ell]) mu = -mu;
            floating s = left_truncated_normal(R, -mu);
            mu += s*std::sqrt(p2i);
            y[i] = (li[ell] ? mu : -mu);
        }

        gammav.data = gl;
        gsl_linalg_QR_lssolve(&Z, tau, &b, &gammav, r);

        assert(!std::isnan(gammav.data[0]));
    }
    gsl_vector_free(tau);
    gsl_vector_free(r);
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
void lda::lda_uncollapsed::save_model(std::ostream& out) const {
    using std::endl;
    out << N_ << endl;
    out << K_ << endl;
    out << F_ << endl;
    out << L_ << endl;
    out << Nwords_ << endl;
    out << alpha_ << endl;
    out << beta_ << endl;
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
        for (const int* j = counts_idx_[i], *cj = counts_[i]; *j != -1; ++j, ++cj) {
            for (int cji = 0; cji != (*cj); ++cji) {
                out << *j << '\t';
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
        for (const int* j = counts_idx_[i], *cj = counts_[i]; *j != -1; ++j, ++cj) {
            for (int cji = 0; cji != (*cj); ++cji) {
                int t;
                in >> t;
                if (t != *j) {
                    throw "*j was expected\n";
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
    out << Ga_ << endl;
    out << Gb_ << endl;
    out << Gn0_ << endl;
    out << Gmu_ << endl;
    out << endl;
    out << endl;
    out << "NOT IMPLEMENTED! Go Away\n\n";
    throw "Not implemented. GO AWAY\n";
}

void lda::lda_collapsed::load_model(std::istream& out) {
    throw "Not implemented.\n";
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
int lda::lda_base::retrieve_gamma(int ell, float* res, int size) const {
    if (size != K_) return 0;
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

int lda::lda_uncollapsed::project_one(const std::vector<int>& words, const std::vector<float>& fs, float* res, int size) {
    if (size != K_) return 0;
    floating thetas[K_];
    this->sample_one(words, fs, thetas);
    std::copy(thetas, thetas + K_, res);
    return size;
}

int lda::lda_collapsed::project_one(const std::vector<int>& words, const std::vector<float>& fs, float* res, int size) {
    if (size != K_) return 0;
    if (int(fs.size()) != F_) return -1;
    if ((*std::max_element(words.begin(), words.end()) >= Nwords_) ||
        (*std::min_element(words.begin(), words.end()) < 0)) return -2;
    floating thetas[K_];
    std::fill(thetas, thetas, alpha_);
    std::vector<int> zs;
    zs.reserve(words.size() + fs.size());
    int counts[K_];
    std::fill(counts, counts + K_, 0);
    for (unsigned zi = 0; zi != (words.size() + fs.size()); ++zi) {
        const int z = R.random_int(0,K_);
        zs.push_back(z);
        ++counts[z];
    }
    for (int it = 0; it != 30; ++it) {
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
            const int k = sample_function(R, p, K_);
            assert(k < K_);
            zs[zi] = k;
            ++counts[k];
        }
    }
    for (int k = 0; k != K_; ++k) res[k] = floating(counts[k])/zs.size();
    assert(std::accumulate(counts, counts + K_, 0) == zs.size());
    return size;
}

float lda::lda_base::score_one(int ell, const float* res, int size) const {
    if (ell >= L_) return -1;
    if (size != K_) return 0;
    return dot_product(res, gamma(ell), size);
}

floating lda::lda_uncollapsed::logperplexity(const std::vector<int>& words, const std::vector<float>& fs) {
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
    return logp;
}

void lda::lda_uncollapsed::sample_one(const std::vector<int>& words, const std::vector<float>& fs, floating* thetas) {
    const int nr_iters = 20;
    std::fill(thetas, thetas + K_, 1.);

    floating crossed[K_ * Nwords_];
    floating offset[Nwords_];
    floating normals[F_][K_];
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
            int z = categorical_sample(R, p, K_);
            ++proposal[z];
        }
        for (int f = 0; f != F_; ++f) {
            floating p[K_];
            for (int k = 0; k != K_; ++k) {
                p[k] = thetas[k] * normals[f][k];
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



