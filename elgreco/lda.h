#ifndef LDA_H_INCLUDE_GUARD_LPC_ELGRECO_
#define LDA_H_INCLUDE_GUARD_LPC_ELGRECO_
#include <vector>
#include <iostream>
#include <cmath>
#include <fstream>

#include "elgreco_random.h"

typedef double floating;

namespace lda {
struct lda_parameters {
    unsigned seed;
    int nr_topics;
    int nr_labels;
    floating alpha;
    floating beta;
    floating lam;
    std::vector<int> area_markers;
};



struct lda_data {
    public:
        lda_data()
            :nr_terms_(0)
            { }
        int nr_words() const {
            int res = 0;
            for (unsigned int i = 0; i != docs_.size(); ++i) res += size(i);
            return res;
        }
        void push_back_doc(const std::vector<int>& nd, const std::vector<floating>& nf, const std::vector<floating>& nl) {
            docs_.push_back(nd);
            for (unsigned i = 0; i != nd.size(); ++i) {
                if (nd[i] >= nr_terms_) nr_terms_ = nd[i]+1;
            }
            features_.push_back(nf);
            labels_.push_back(nl);
        }

        std::vector<int>& at(int d) { return docs_[d]; }
        int nr_docs() const { return docs_.size(); }
        int size(int d) const { return docs_[d].size(); }
        int operator()(int d, int w) const { return docs_.at(d).at(w); }
        int nr_terms() const { return nr_terms_; }

        std::vector<floating>& features_at(int d) { return features_.at(d); }
        floating feature(int d, int w) const { return features_[d][w]; }
        int nr_features() const { return features_[0].size(); }

        std::vector<floating>& labels_at(int d) { return labels_.at(d); }
        int nr_labels() const { assert(!labels_.empty()); return labels_[0].size(); }
        floating label(int d, int ell) const { return labels_[d][ell]; }
    private:
        std::vector< std::vector<int> > docs_;
        std::vector< std::vector<floating> > features_;
        std::vector< std::vector<floating> > labels_;
        int nr_terms_;
};

lda_data load(std::istream &in);

struct lda_base {
    public:
        lda_base(lda_data& data, lda_parameters params);
        virtual ~lda_base() {
            delete [] words_[0];
            delete [] words_;
            delete [] features_[0];
            delete [] features_;
            delete [] ls_;
            delete [] gamma_;
        }
        // One sampling step
        virtual void step() = 0;

        // Forward sampling.
        // This should be performed to initialize sampler to good values.
        virtual void forward() = 0;

        // log likelihood of model
        virtual floating logP(bool normalise=false) const = 0;

        virtual int retrieve_theta(int i, float* res, int size) const = 0;

        // Print is human readable
        virtual void print_topics(std::ostream&) const = 0;
        virtual void print_words(std::ostream&) const = 0;

        // Model saving/loading is mostly human-unreadable
        virtual void save_model(std::ostream&) const = 0;
        // Shortcut: saves to file
        void save_to(const char* fname) const {
            std::ofstream out(fname);
            save_model(out);
        }

        virtual void load_model(std::istream&) = 0;
        // Shortcut: loads from file
        void load_from(const char* fname) {
            std::ifstream in(fname);
            load_model(in);
        }

        virtual int project_one(const std::vector<int>&, const std::vector<float>&, float* res, int size) const = 0;
        virtual floating logperplexity(const std::vector<int>& ws, const std::vector<float>& fs, const std::vector<float>& labels) const = 0;
        int nr_docs() const { return N_; }
        int nr_topics() const { return K_; }
        int nr_labels() const { return L_; }

        int retrieve_gamma(int ell, float* res, int size) const;
        // Returns strength of ell label for [array]
        // (for supervised models only)
        float score_one(int ell, const float* array, int size) const;

    protected:
        mutable random_source R;
        int K_;
        int N_;
        int F_;
        int L_;
        int Nwords_;

        struct sparse_int {
            int value;
            int count;
            sparse_int()
                :value(-2)
                ,count(-2)
                { }
            sparse_int(int v, int c)
                :value(v)
                ,count(c)
                { }
        };
        sparse_int** words_;

        floating** features_;

        floating* ls_;
        floating* ls(int i) { return ls_ + i*L_; }
        const floating* ls(int i) const { return ls_ + i*L_; }

        floating* gamma_;
        floating* gamma(int ell) {assert(ell <= L_); return gamma_ + ell*K_; }
        const floating* gamma(int ell) const { assert(ell <= L_); return gamma_ + ell*K_; }

        floating alpha_;
        floating beta_;
        floating lambda_;

        floating Ga_;
        floating Gb_;
        floating Gn0_;
        floating Gmu_;
};


struct lda_uncollapsed : lda_base {
    public:
        lda_uncollapsed(lda_data& data, lda_parameters params);
        ~lda_uncollapsed() {
            delete [] multinomials_[0];
            delete [] multinomials_;
            delete [] thetas_;
            delete [] normals_[0];
            delete [] normals_;
            delete [] sample_;

            if (zs_) {
                for (int i = 0; i != N_; ++i) {
                    delete [] zs_[i];
                }
                delete [] zs_;
            }
            delete [] z_bars_;

            delete [] ys_;
        }
        virtual void step();
        virtual void forward();
        virtual floating logP(bool normalise=false) const;
        void load(std::istream& topics, std::istream& words);

        int retrieve_logbeta(int k, float* res, int size) const;
        virtual int retrieve_theta(int i, float* res, int size) const;
        int retrieve_ys(int i, float* res, int size) const;
        int retrieve_z_bar(int i, float* res, int size) const;

        int set_logbeta(int k, float* res, int size);
        int set_theta(int i, float* res, int size);

        int project_one(const std::vector<int>&, const std::vector<float>&, float* res, int size) const;
        void nosample(int i) { sample_[i] = false; }
        void sample(int i) { sample_[i] = true; }

        virtual floating logperplexity(const std::vector<int>& ws, const std::vector<float>& fs, const std::vector<float>& labels) const;

        void print_topics(std::ostream&) const;
        void print_words(std::ostream&) const;
        void save_model(std::ostream&) const;
        void load_model(std::istream&);

    private:
        void sample_one(const std::vector<int>&, const std::vector<float>&, floating*) const;

        floating** multinomials_;
        normal_params** normals_;

        floating* thetas(int i) { return thetas_ + i*K_; }
        const floating* thetas(int i) const { return thetas_ + i*K_; }
        floating* thetas_;

        bool* sample_;

        floating* z_bar(int i) { return z_bars_ + i*K_; }
        const floating* z_bar(int i) const { return z_bars_ + i*K_; }
        floating* z_bars_;

        int** zs_;

        floating* ys_;
        floating* ys(int i) { return ys_ + i*L_; }
        const floating* ys(int i) const { return ys_ + i*L_; }

};

struct lda_collapsed : lda_base {
    public:
        lda_collapsed(lda_data& data, lda_parameters params);
        ~lda_collapsed() {
        }
        virtual void step();
        virtual void forward();
        virtual floating logP(bool normalise=false) const;

        virtual int retrieve_theta(int i, float* res, int size) const;

        int project_one(const std::vector<int>&, const std::vector<float>&, float* res, int size) const;
        virtual floating logperplexity(const std::vector<int>& ws, const std::vector<float>& fs, const std::vector<float>& labels) const;

        void print_topics(std::ostream&) const;
        void print_words(std::ostream&) const;

        void save_model(std::ostream&) const;
        void load_model(std::istream&);

        lda_data get_data() const;
        lda_parameters get_parameters() const;

        // This function verifies a few invariants.
        // It assert()s many things that should always be true
        void verify() const;
    private:
        void sample_one(const std::vector<int>&, const std::vector<float>&, std::vector<int>&, floating counts[]) const;
        void update_gammas();
        void update_alpha_beta();

        int area_of(int w) const {
            for (unsigned a = 0; a != area_markers_.size(); ++a) {
                if (w < area_markers_[a]) return a;
            }
            return area_markers_.size();
        }
        int area_size(int a) const {
            if (a == 0) return area_markers_[0];
            return area_markers_.at(a) - area_markers_[a-1];
        }

        typedef std::vector<int>::iterator vint_iter;
        typedef std::vector<floating>::iterator vfloating_iter;
        typedef std::vector<floating>::const_iterator vfloating_c_iter;

        std::vector<int> zidata_;
        std::vector<vint_iter> zi_;

        std::vector<int> area_markers_;
        int nr_areas_;

        std::vector<floating> topic_;
        std::vector<floating> topic_count_data_;
        vfloating_iter topic_count(const int i) { return topic_count_data_.begin() + i *K_; }
        vfloating_c_iter topic_count(const int i) const { return topic_count_data_.begin() + i *K_; }

        std::vector<floating> topic_area_data_;
        vfloating_iter topic_area(const int a) { return topic_area_data_.begin() + a*K_; }
        vfloating_c_iter topic_area(const int a) const { return topic_area_data_.begin() + a*K_; }

        vfloating_iter topic_term(const int t) { return topic_term_data_.begin() + t*K_; }
        vfloating_c_iter topic_term(const int t) const { return topic_term_data_.begin() + t*K_; }
        std::vector<floating> topic_term_data_;

        vfloating_iter topic_numeric_count(const int f) { return topic_numeric_count_data_.begin() + f*K_; }
        vfloating_c_iter topic_numeric_count(const int f) const { return topic_numeric_count_data_.begin() + f*K_; }
        std::vector<floating> topic_numeric_count_data_;

        int size(int i) const { assert(i < N_); return zi_[i+1]-zi_[i]; }

        vfloating_iter sum_f(int f) { assert(f < F_); return sum_f_.begin() + f*K_; }
        vfloating_c_iter sum_f(int f) const { assert(f < F_); return sum_f_.begin() + f*K_; }

        vfloating_iter sum_f2(int f) { assert(f < F_); return sum_f2_.begin() + f*K_; }
        vfloating_c_iter sum_f2(int f) const { assert(f < F_); return sum_f2_.begin() + f*K_; }

        std::vector<floating> sum_f_;
        std::vector<floating> sum_f2_;
};

}
#endif // LDA_H_INCLUDE_GUARD_LPC_ELGRECO_
