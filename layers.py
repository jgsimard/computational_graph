# -*- coding: utf-8 -*-
"""
@author: Jean-Gabriel Simard
"""

import numpy as np
from graph import Parameter
import operations as op

def dnn(X, n_in, n_out):
    W = Parameter(np.random.randn(n_in, n_out))
    b = Parameter(np.random.randn(n_out))
    p = op.sigmoid(op.add(op.matmul(X, W), b))
    return p