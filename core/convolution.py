# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 12:06:02 2018

@author: Jean-Gabriel Simard
"""
from core.graph import Operation

class Convolution2DNaive(Operation):
    def __init__(self, x, weights):
        super().__init__([x, weights])
        self.name = 'Convolution2DNaive'
        
class Convolution1DNaive(Operation):
    def __init__(self, x, weights):
        super().__init__([x, weights])
        self.name = 'Convolution2DNaive'