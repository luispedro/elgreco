#ifndef LPC_ELGRECO_RANDOM_H_THU_APR_28_22_16_31_EDT_2011
#define LPC_ELGRECO_RANDOM_H_THU_APR_28_22_16_31_EDT_2011

#include <iostream>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf_gamma.h>
#include <cmath>
#include <assert.h>
typedef double floating;

struct random_source {
    random_source(unsigned s=1234567890)
        :r(gsl_rng_alloc(gsl_rng_mt19937))
    {
        gsl_rng_set(r, s);
    }
    random_source(const random_source& rhs)
        :r(gsl_rng_clone(rhs.r))
        {
        }
    ~random_source() {
        gsl_rng_free(r);
    }


    random_source& operator = (const random_source& rhs) {
        random_source n(rhs);
        this->swap(n);
        return *this;
    }

    void swap(random_source& rhs) {
        std::swap(r, rhs.r);
    }

    floating uniform01() {
        return gsl_ran_flat(r, 0., 1.);
    }
    floating gamma(floating a, floating b) {
        return gsl_ran_gamma(r, a, b);
    }
    floating normal(floating mu, floating sigma) {
        return mu + gsl_ran_gaussian(r, sigma);
    }
    floating bayesian_normal(floating mu, floating precision) {
        return mu + gsl_ran_gaussian(r, 1./std::sqrt(precision));
    }
    floating exponential(floating lam) {
        return gsl_ran_exponential(r, lam);
    }
    int random_int(int a, int b) {
        return int(gsl_ran_flat(r, double(a),double(b)));
    }
    private:
       gsl_rng * r;
};

struct normal_params {
    normal_params(floating mu, floating precision)
        :mu(mu)
        ,precision(precision)
        { }
    normal_params() { }
    floating mu;
    floating precision;
};

const floating _pi = 3.1415926535898;
const floating _inv_two_pi = 1./std::sqrt(2. * _pi);

inline
floating normal_like(const floating value, const normal_params& params, bool normalise=true) {
    floating d = (value - params.mu) * params.precision;
    return _inv_two_pi * params.precision * std::exp( -d*d );
}

inline
normal_params normal_gamma(random_source& R, const floating mu0, const floating kappa0, const floating alpha, const floating beta) {
    assert(beta != 0.);
    const floating tao = R.gamma(alpha, 1./beta);
    assert(tao > 0);
    const floating mu = (kappa0 * tao < 100 ? R.bayesian_normal(mu0, kappa0 * tao): mu0);
    return normal_params(mu, tao);
}

inline
floating left_truncated_normal(random_source& R, const floating mu) {
    assert(!std::isnan(mu));
    // For an implementation reference, see
    // Simulation of truncated normal variables
    // by Christian P. Robert
    //
    // http://arxiv.org/abs/0907.4010
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
            const floating z = mu + R.exponential(1./alphastar);
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
#endif 

