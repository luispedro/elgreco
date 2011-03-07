%module lda

%include "std_vector.i"
namespace std {
    %template(vectori) std::vector<int>;
}

%{
#define SWIG_FILE_WITH_INIT
#include "lda.h"
%}


%include "lda.h"

