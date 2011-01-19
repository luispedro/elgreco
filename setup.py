# -*- coding: utf-8 -*-
# Copyright (C) 2011, Luis Pedro Coelho <lpc@cmu.edu>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
# 
# LICENSE: GPLv3

from __future__ import division
from sys import exit
try:
    import setuptools
except:
    print '''
setuptools not found. Please install it.

On linux, the package is often called python-setuptools'''
    exit(1)

execfile('elgreco/elgreco_version.py')
long_description = file('README.rst').read()

classifiers = [
'Environment :: Console',
'License :: OSI Approved :: MIT License',
'Operating System :: POSIX',
'Operating System :: OS Independent',
'Programming Language :: Python',
'Topic :: Scientific/Engineering',
'Topic :: Software Development',
'Intended Audience :: Science/Research',
]

setuptools.setup(name='elgreco',
      version=__version__,
      description='El Graphical Model Compiler',
      long_description=long_description,
      author='Luis Pedro Coelho',
      author_email='luis@luispedro.org',
      license='GPLv3',
      platforms=['Any'],
      classifiers=classifiers,
      url='http://luispedro.org/software/elgreco',
      packages=setuptools.find_packages(exclude='tests'),
      test_suite='nose.collector',
      )


