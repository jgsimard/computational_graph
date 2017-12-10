# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 02:41:51 2017

@author: Jean-Gabriel Simard
"""

import numpy as np
import matplotlib.pyplot as plt
import graph, loss, layers, gradient_descent, utils, operations
from graph import Operation



def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))   

x_shape = (2, 3, 4, 4)
w_shape = (3, 3, 4, 4)
x = np.linspace(-0.1, 0.5, num=np.prod(x_shape)).reshape(x_shape)
w = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape)
b = np.linspace(-0.1, 0.2, num=3)

X = graph.Placeholder(name = 'inputs') #to feed with attributes
W = graph.Parameter(w)
B = graph.Parameter(b)

#def __init__(self, x, channel_in, channel_out, kernel_shape = None, padding = 1, stride = 1):
#op = operations.convolution(X,W,stride = 2, padding = 1)
#out = op + B

op = operations.Convolution(X,W,B, stride = 2)
#out, cache = conv_forward(x, w, b, stride=2, padding=1)

#utils.draw_graph(op.graph)

correct_out = np.array([[[[-0.08759809, -0.10987781],
                           [-0.18387192, -0.2109216 ]],
                          [[ 0.21027089,  0.21661097],
                           [ 0.22847626,  0.23004637]],
                          [[ 0.50813986,  0.54309974],
                           [ 0.64082444,  0.67101435]]],
                         [[[-0.98053589, -1.03143541],
                           [-1.19128892, -1.24695841]],
                          [[ 0.69108355,  0.66880383],
                           [ 0.59480972,  0.56776003]],
                          [[ 2.36270298,  2.36904306],
                           [ 2.38090835,  2.38247847]]]])

session = graph.Session()
session.run(op,{X: x})

#print(rel_error(correct_out,op.output))

