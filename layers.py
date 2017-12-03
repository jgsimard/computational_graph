# -*- coding: utf-8 -*-
"""
@author: Jean-Gabriel Simard
"""

import numpy as np
from graph import Parameter
from operations import sigmoid, softmax, add, matmul

def dnn(X, n_in, n_out, activation = None):
    W = Parameter(np.random.randn(n_in, n_out))
    b = Parameter(np.random.randn(n_out))
    temp = add(matmul(X, W), b)
    
    if activation == 'softmax':
        return softmax(temp)
    elif activation == 'sigmoid':
        return sigmoid(temp)
    else:
        return sigmoid(temp)