# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 22:40:26 2018

@author: Jean-Gabriel Simard
"""
from core.graph import Operation
import numpy as np

class MaxPooling(Operation):
    def __init__(self, x, stride = 1, pool_size = 2):
        super().__init__([x])
        self.name = "Pooling"
        self.stride = stride
        self.pool_size = pool_size
    
    def compute (self, x_value):
        n_input,  x_c, x_h, x_w = x_value.shape
        out_h = (x_h - self.pool_size) / self.stride + 1
        out_w = (x_w - self.pool_size) / self.stride + 1
        out = np.zeros((n_input,x_c, out_h, out_w))
        
        for i in range(n_input):
            for c in range(x_c):
                for h in range(0,x_h,self.stride):
                    for w in range(0, x_w, self.stride):
                        
                        out[i, c, h/self.stride, w/self.stride] = np.max(x_value[i, c, h:h+self.pool_size, w:w+self.pool_size])
        return out
    
    def gradient(self, grad):
        n_input,  x_c, x_h,   x_w   = self.input_nodes[0].output.shape
        n_input,  x_c, out_h, out_w = self.output.shape
        dx = np.zeros(self.input_nodes[0].output.shape)
        
        for i in range(n_input):
            for c in range(x_c):
                for h in range(0,x_h,self.stride):
                    for w in range(0, x_w, self.stride):
                        pool = self.input_nodes[0].output[i, c, h:h+self.pool_size, w:w+self.pool_size]
                        mask = pool == np.max(pool) 
                        dx[i, c, h:h+self.pool_size, w:w+self.pool_size] = mask * grad[i,c,h,w]