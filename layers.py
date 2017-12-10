# -*- coding: utf-8 -*-
"""
@author: Jean-Gabriel Simard
"""
import numpy as np
from graph import Parameter
from operations import sigmoid, softmax, matmul, leaky_relu, Convolution


#x_shape = (2, 3, 4, 4)
#w_shape = (3, 3, 4, 4)
#x = np.linspace(-0.1, 0.5, num=np.prod(x_shape)).reshape(x_shape)
#w = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape)
#b = np.linspace(-0.1, 0.2, num=3)
#
#X = graph.Placeholder(name = 'inputs') #to feed with attributes
#W = graph.Parameter(w)
#B = graph.Parameter(b)


def convolutional(X, channel_in, channel_out, filter_height = 3, filter_width = 3, stride = 1, pad = 1):
    W = Parameter(np.random.randn(channel_out, channel_in,filter_height, filter_width), 'Weights')
    b = Parameter(np.random.randn(channel_out), 'Bias')
    return Convolution(X, W, b, stride, pad)
    
    
def fully_connected(X, n_in, n_out, activation = None):
    W = Parameter(np.random.randn(n_in, n_out), 'Weights')
    b = Parameter(np.random.randn(n_out), 'Bias')
    temp = matmul(X, W) + b    
    
    if activation == 'softmax':
        return softmax(temp)
    
    elif activation == 'sigmoid':
        return sigmoid(temp)
    
    else:
        return leaky_relu(temp)
    
def perceptron(X, n_in, n_out):
    W = Parameter(np.random.randn(n_in, n_out), 'Weights')
    b = Parameter(np.random.randn(n_out), 'Bias')
    return matmul(X, W) + b    
