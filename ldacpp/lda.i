%module lda

%include "std_vector.i"
namespace std {
    %template(vectori) std::vector<int>;
}
%{
#define SWIG_FILE_WITH_INIT
%}
%include "numpy.i"
%init %{
    import_array();
%}

%apply (float* INPLACE_ARRAY1, int DIM1) {(float* res, int size)}

%{
#include "lda.h"
%}


%include "lda.h"

