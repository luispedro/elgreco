#ifndef LPC_ELGRECO_RANDOM_H_THU_APR_28_22_16_31_EDT_2011
#define LPC_ELGRECO_RANDOM_H_THU_APR_28_22_16_31_EDT_2011

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf_gamma.h>
#include <cmath>
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
#endif 

