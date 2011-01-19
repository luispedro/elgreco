# Copyright (C) 2011, Luis Pedro Coelho <lpc@cmu.edu>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
# 
# LICENSE: GPLv3

from __future__ import division
import numpy as np
class ConstantModel(object):
    def __init__(self, value):
        self.value = value

    def logP(self, value, parents):
        if value == self.value: return 0.
        return float("-Inf")

    def sample1(self, _, __):
        return self.value
