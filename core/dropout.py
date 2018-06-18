# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 16:36:54 2018

@author: Jean-Gabriel Simard
"""

import numpy as np
from core.graph import Operation

class Dropout(Operation):
    def __init__(self, x, dropout):
        super().__init__([x])
        self.name = "Dropout"

    def compute(self, x_value, dropout_value):
        if(dropout_value["mode"] == "train"):
            self.mask =  (np.random.rand(*x_value.shape) < dropout_value['p']) / dropout_value['p']
            return x_value * self.mask
        return x_value
    
    def gradient(self,  grad): #only used during backprop
        return [grad * self.mask, 0.0]
        