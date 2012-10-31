%exception {
    try {
        $action
    } catch (const char* msg) {
        PyErr_SetString(PyExc_RuntimeError, msg);
        return NULL;
    } catch (std::string msg) {
        PyErr_SetString(PyExc_RuntimeError, msg.c_str());
        return NULL;
    }
}

%module lda
%include "std_vector.i"
%include "std_string.i"
namespace std {
    %template(vectori) std::vector<int>;
    %template(vectord) std::vector<double>;
    %template(vectorf) std::vector<float>;
    %template(vectorb) std::vector<bool>;
}
%{
#define SWIG_FILE_WITH_INIT
%}
%include "numpy.i"
%init %{
    import_array();
%}

%apply (float* INPLACE_ARRAY1, int DIM1) {(float* res, int size)}
%apply (float* IN_ARRAY1, int DIM1) {(const float* array, int size)}

%{
#include <sstream>
#include "lda.h"

std::string save_model_to_string(const lda::lda_base& lda) {
    std::stringstream out;
    lda.save_model(out);
    return out.str();
}

void load_model_from_string(lda::lda_base& lda, const std::string& input) {
    std::stringstream in(input);
    lda.load_model(in);
}
%}
%include "lda.h"

std::string save_model_to_string(const lda::lda_base& lda);
void load_model_from_string(lda::lda_base& lda, const std::string& input);

