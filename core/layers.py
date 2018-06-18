# -*- coding: utf-8 -*-
"""
@author: Jean-Gabriel Simard
"""
import numpy as np
from core.graph import Parameter, scope
from core.operations import Sigmoid, Softmax, Matmul, Leaky_relu
import convolution


def convolutional_naive(X, channel_in, channel_out, filter_height = 3, filter_width = 3, stride = 1, pad = 1):
    W = Parameter(np.random.randn(channel_out, channel_in,filter_height, filter_width), 'Weights')
    b = Parameter(np.random.randn(channel_out), 'Bias')
    return convolution.Convolution2DNaive(X, W, b, stride, pad)
        
def convolutional(X, channel_in, channel_out, filter_height = 3, filter_width = 3, stride = 1, pad = 1):
    W = Parameter(np.random.randn(channel_out, channel_in,filter_height, filter_width), 'Weights')
    b = Parameter(np.random.randn(channel_out), 'Bias')
    return convolution.Convolution(X, W, b, stride, pad)
    
    
def fully_connected(X, n_in, n_out, activation = None, name = None):
    with scope('fully_connected', X.graph): # Not working yet
        W = Parameter(np.random.randn(n_in, n_out), 'Weights')
        b = Parameter(np.random.randn(n_out), 'Bias')
        temp = Matmul(X, W) + b    
        
        if activation == 'softmax':
            return Softmax(temp)
        
        elif activation == 'sigmoid':
            return Sigmoid(temp)
        
        elif activation == 'linear':
            return temp
    
        else:
            return Leaky_relu(temp)
