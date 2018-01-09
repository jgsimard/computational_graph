# -*- coding: utf-8 -*-
"""
@author: Jean-Gabriel Simard
"""
import numpy as np
from core.graph import Parameter
from core.operations import Sigmoid, Softmax, Matmul, Leaky_relu, Convolution

from copy import deepcopy #not the most elegent code

class scope():

    def __init__(self, name, graph):
        self.name  = name
        self.graph = graph
        self.graph_before = deepcopy(graph)

    def __enter__(self):
        pass

    def __exit__(self, *args):
        for op in self.graph.operations:
            if op not in self.graph_before.operations:
                op.name = self.name + '/' + op.name
                
        for p in self.graph.placeholders:
            if p not in self.graph_before.placeholders:
                p.name = self.name + '/' + p.name
                
        for par in self.graph.parameters:
            if par not in self.graph_before.parameters:
                par.name = self.name + '/' + par.name
        
def convolutional(X, channel_in, channel_out, filter_height = 3, filter_width = 3, stride = 1, pad = 1):
    W = Parameter(np.random.randn(channel_out, channel_in,filter_height, filter_width), 'Weights')
    b = Parameter(np.random.randn(channel_out), 'Bias')
    return Convolution(X, W, b, stride, pad)
    
    
def fully_connected(X, n_in, n_out, activation = None, name = None):
#    with scope('fully_connected', X.graph): Not working yet
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
