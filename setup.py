# -*- coding: utf-8 -*-
# Copyright (C) 2011, Luis Pedro Coelho <luis@luispedro.org>

from distutils.core import setup, Extension
import os

undef_macros=[]
if os.environ.get('DEBUG'):
    undef_macros=['NDEBUG']

lda_module = Extension(
                'elgreco._lda',
                sources=['elgreco/lda.i', 'elgreco/lda.cpp', 'elgreco/load.cpp'],
                libraries=['gsl', 'gslcblas'],
                extra_compile_args=['-fopenmp'],
                extra_link_args=['-lgomp'],
                undef_macros=undef_macros,
                swig_opts=['-c++'],
               )

random_module = Extension(
                'elgreco.elgreco_random',
                sources=['elgreco/elgreco_random.i'],
                swig_opts=['-c++'],
                )

setup (name = 'elgreco',
       version = '0.1',
       author = 'Luis Pedro Coelho <luis@luispedro.org>',
       ext_modules = [lda_module, random_module],
       py_modules = ['elgreco.lda'],
       )
