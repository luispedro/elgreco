# -*- coding: utf-8 -*-
# Copyright (C) 2011-2012, Luis Pedro Coelho <luis@luispedro.org>

from setuptools import setup, Extension
import os

undef_macros = []
define_macros = []
if os.environ.get('DEBUG'):
    undef_macros = ['NDEBUG']
    if os.environ.get('DEBUG') == '2':
        define_macros = [('_GLIBCXX_DEBUG','1')]


execfile('elgreco/elgreco_version.py')

lda_module = Extension(
                'elgreco._lda',
                sources=['elgreco/lda.i', 'elgreco/lda.cpp', 'elgreco/load.cpp'],
                libraries=['gsl', 'gslcblas'],
                extra_compile_args=['-fopenmp', '-funsafe-math-optimizations', '-ffast-math'],
                extra_link_args=['-lgomp'],
                undef_macros=undef_macros,
                define_macros=define_macros,
                swig_opts=['-c++'],
               )

random_module = Extension(
                'elgreco._elgreco_random',
                sources=['elgreco/elgreco_random.i'],
                undef_macros=undef_macros,
                define_macros=define_macros,
                libraries=['gsl', 'gslcblas'],
                swig_opts=['-c++'],
                )

setup (name = 'elgreco',
       version = __version__,
       author = 'Luis Pedro Coelho <luis@luispedro.org>',
       ext_modules = [lda_module, random_module],
       py_modules = ['elgreco.lda', 'elgreco.ldahelper'],
       test_suite = 'nose.collector',
       )
