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
     * data = [phi(i/128.) for i in xrange(256)]
     * print data
     */
    const floating table[] = {
        0.5, 0.50440764144758954, 0.50881474489132095, 0.51322077252433684, 0.51762518693366144, 0.52202745129684047, 0.52642702957822185, 0.53082338672475526, 0.53521598886119359, 0.539604303484576, 0.54398779965787614, 0.54836594820269569, 0.55273822189088984, 0.55710409563500585, 0.56146304667742319, 0.56581455477807874, 0.57015810240066689, 0.57449317489720308, 0.57881926069084011, 0.58313585145683, 0.58744244230152498, 0.59173853193931203, 0.59602362286737787, 0.60029722153820331, 0.60455883852968795, 0.60880798871280561, 0.61304419141669875, 0.61726697059111479, 0.62147585496609592, 0.62567037820883253, 0.62985007907759383, 0.63401450157265193, 0.63816319508411845, 0.64229571453661283, 0.64641162053068868, 0.65051047948094154, 0.6545918637507302, 0.65865535178344015, 0.66270052823022629, 0.66672698407417075, 0.67073431675079753, 0.67472213026488559, 0.67869003530352812, 0.68263764934538607, 0.68656459676608883, 0.69047050893973783, 0.69435502433646989, 0.69821778861604344, 0.70205845471741113, 0.70587668294424744, 0.70967214104640086, 0.71344450429724593, 0.71719345556691194, 0.72091868539136772, 0.72461989203734711, 0.72829678156310118, 0.7319490678749665, 0.73557647277974314, 0.73917872603287771, 0.74275556538245047, 0.74630673660896907, 0.74983199356097296, 0.75333109818645894, 0.75680382056013551, 0.76024993890652326, 0.76366923961891708, 0.76706151727422989, 0.77042657464374042, 0.7737642226997723, 0.77707428061833062, 0.78035657577772877, 0.7836109437532377, 0.78683722830779601, 0.7900352813788174, 0.79320496306113952, 0.79634614158615524, 0.79945869329717545, 0.80254250262106974, 0.80559746203623828, 0.80862347203696483, 0.81162044109420894, 0.81458828561289431, 0.81752692988575004, 0.82043630604377094, 0.82331635400335412, 0.82616702141018261, 0.82898826357991839, 0.83178004343577516, 0.83454233144304069, 0.83727510554062001, 0.83997835106967156, 0.84265206069941212, 0.84529623435016332, 0.84791087911372065, 0.8504960091711169, 0.85305164570786562, 0.8555778168267576, 0.85807455745829631, 0.86054190926885143, 0.86297992056661199, 0.86538864620542366, 0.86776814748659248, 0.87011849205873959, 0.87243975381578986, 0.87473201279318102, 0.87699535506237658, 0.87922987262376817, 0.88143566329805312, 0.8836128306161708, 0.88576148370788432, 0.8878817371890928, 0.88997371104795797, 0.89203753052992973, 0.89407332602175726, 0.89608123293456554, 0.89806139158608556, 0.90001394708211646, 0.90193904919730583, 0.90383685225532773, 0.90570751500853974, 0.90755120051719984, 0.90936807602832304, 0.91115831285425419, 0.91292208625103755, 0.91465957529665753, 0.91637096276922692, 0.91805643502519829, 0.91971618187766913, 0.92135039647485739, 0.92295927517881382, 0.92454301744444378, 0.92610182569890653, 0.92763590522145845, 0.92914546402380693, 0.93063071273103781, 0.93209186446318126, 0.93352913471747634, 0.9349427412513942, 0.9363329039664785, 0.9376998447930609, 0.93904378757590501, 0.94036495796083464, 0.94166358328239697, 0.94293989245261134, 0.94419411585085389, 0.94542648521492278, 0.94663723353333196, 0.94782659493887622, 0.94899480460350882, 0.95014209863457433, 0.95126871397243273, 0.95237488828951511, 0.95346085989084317, 0.95452686761604788, 0.95557315074291882, 0.95659994889251521, 0.95760750193586608, 0.95859604990228686, 0.95956583288933794, 0.96051709097444871, 0.96145006412822909, 0.96236499212948745, 0.96326211448197618, 0.96414167033287934, 0.96500389839305978, 0.96584903685907775, 0.96667732333699585, 0.96748899476797812, 0.96828428735569438, 0.96906343649553794, 0.96982667670566114, 0.97057424155983574, 0.97130636362213929, 0.97202327438347147, 0.97272520419990083, 0.97341238223283932, 0.97408503639104516, 0.97474339327444959, 0.97538767811980231, 0.97601811474813194, 0.97663492551401143, 0.97723833125662418, 0.97782855125261769, 0.97840580317073789, 0.97897030302822996, 0.97952226514899599, 0.98006190212349231, 0.98058942477035649, 0.98110504209974558, 0.98160896127837016, 0.98210138759620791, 0.98258252443487737, 0.98305257323765538, 0.98351173348111565, 0.98396020264837125, 0.98439817620389858, 0.98482584756992109, 0.98524340810433175, 0.98565104708013007, 0.98604895166635198, 0.98643730691046683, 0.98681629572221841, 0.98718609885888542, 0.98754689491193548, 0.9878988602950457, 0.98824216923346764, 0.9885769937547062, 0.98890350368048807, 0.9892218666199919, 0.98953224796431405, 0.98983481088214109, 0.99012971631660252, 0.99041712298327544, 0.99069718736931389, 0.99097006373367336, 0.99123590410840512, 0.99149485830098905, 0.9917470738976788, 0.9919926962678316, 0.99223186856919243, 0.99246473175410554, 0.99269142457662718, 0.99291208360050742, 0.99312684320801725, 0.99333583560959116, 0.99353919085425779, 0.99373703684083159, 0.99392949932983821, 0.99411670195614765, 0.99429876624228464, 0.99447581161239584, 0.99464795540684103, 0.99481531289738756, 0.99497799730297865, 0.99513611980605199, 0.99528978956938374, 0.9954391137534313, 0.99558419753415184, 0.99572514412127244, 0.99586205477698742, 0.99599502883505997, 0.99612416372030421, 0.99624955496842649, 0.99637129624620224, 0.99648947937196741, 0.99660419433640213, 0.9967155293235872, 0.99682357073231098, 0.99692840319760667, 0.99703010961250249, 0.99712877114996257, 0.99722446728500014, 0.99731727581694707, 0.9974072728918586, 0.99749453302503677, 0.99757912912365643 };

    assert(!std::isnan(x));
    if (x < 0) return 1.-phi(-x);
    const int idx = int(x * 128);
    if (idx >= 256) return table[255];
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
        zi_[0] = z_;
        int* zinext = z_;
        for (int i = 0; i != N_; ++i) {
            int c = 0;
            for (const sparse_int* j = words_[i]; j->value != -1; ++j) {
                for (int cji = 0; cji != j->count; ++cji) {
                    ++zinext;
                    ++c;
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
    floating zb_gamma[L_];
    for (int i = 0; i != N_; ++i) {
        floating z_bar[K_];
        std::fill(z_bar, z_bar + K_, 0);
        const int Ni = size(i);
        for (int d = 0; d != Ni; ++d) {
            ++z_bar[z[d]];
        }
        for (int k = 0; k != K_; ++k) z_bar[k] /= Ni;
        for (int ell = 0; ell != L_; ++ell) {
            zb_gamma[ell] = dot_product(z_bar, gamma(ell), K_);
        }

        for (const sparse_int* j = words_[i]; j->value != -1; ++j) {
            for (int cji = 0; cji != j->count; ++cji) {
                floating p[K_];
                const int ok = *z;
                --topic_count_[i][ok];
                --topic_[ok];
                --topic_term_[j->value][ok];
                for (int k = 0; k != K_; ++k) {
                    p[k] = (topic_term_[j->value][k] + beta_)/
                                (topic_[k] + beta_) *
                        (topic_count_[i][k] + alpha_)/
                                (size(i) + alpha_ - 1);
                    for (int ell = 0; ell != L_; ++ell) {
                        const floating delta = gamma(ell)[k] - gamma(ell)[ok];
                        const floating s = ls(i)[ell] ? +1 : -1;
                        p[k] *= phi(s * (zb_gamma[ell]+delta/Ni));
                    }
                    assert(!std::isnan(p[k]));
                }
                const int k = categorical_sample(R, p, K_);

                *z++ = k;
                ++topic_count_[i][k];
                ++topic_[k];
                ++topic_term_[j->value][k];
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
             == size(i));
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
        for (const sparse_int* j = words_[i]; j->value != -1; ++j) {
            for (int cji = 0; cji != j->count; ++cji) {
                const int k = R.random_int(0, K_);
                *z++ = k;
                ++topic_count_[i][k];
                ++topic_term_[j->value][k];
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
    gsl_matrix_view Z = gsl_matrix_view_array(zbars.get(), N_, K_);
    gsl_vector_view b = gsl_vector_view_array(y.get(), N_);

    gsl_vector* r = gsl_vector_alloc(N_);
    gsl_vector* tau = gsl_vector_alloc(K_);

    for (int i = 0; i != N_; ++i) {
        floating* zb = zbars.get() + K_*i;
        for (int k = 0; k != K_; ++k) {
            zb[k] = (topic_count_[i][k] + alpha_)/(size(i) + K_*alpha_);
        }
    }

    gsl_linalg_QR_decomp(&Z.matrix, tau);
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

        gsl_vector_view gammav = gsl_vector_view_array(gl, K_);
        gsl_linalg_QR_lssolve(&Z.matrix, tau, &b.vector, &gammav.vector, r);
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
    const int* z = z_;
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



