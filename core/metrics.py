# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 22:39:30 2018

@author: Jean-Gabriel Simard
"""

import numpy as np
from core import Operation

class Accuracy(Operation)   :
    def __init__(self, x,y):
        super().__init__([x,y])
        self.name = "Accuracy"
    
    def compute(self, x_value, y_value):
        return np.sum(np.argmax(x_value, axis=1)==np.argmax(y_value, axis = 1))/x_value.shape[0]
    
    def gradient(self, grad):
        pass