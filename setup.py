# -*- coding: utf-8 -*-
# Copyright (C) 2011, Luis Pedro Coelho <luis@luispedro.org>

from numpy.distutils.core import setup, Extension

lda_module = Extension(
                'elgreco._lda',
                sources=['elgreco/lda_wrap.cxx', 'elgreco/lda.cpp', 'elgreco/load.cpp'],
                libraries=['gsl', 'gslcblas'],
                extra_compile_args=['-fopenmp'],
                extra_link_args=['-lgomp'],
               )

setup (name = 'elgreco',
       version = '0.1',
       author = 'Luis Pedro Coelho <luis@luispedro.org>',
       ext_modules = [lda_module],
       py_modules = ['elgreco.lda'],
       )
