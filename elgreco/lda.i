%exception {
    try {
        $action
    } catch (const char* msg) {
        PyErr_SetString(PyExc_RuntimeError, msg);
        return NULL;
    }
}

%module lda
%include "std_vector.i"
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
#include "lda.h"
%}


%include "lda.h"


