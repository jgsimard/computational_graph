# -*- coding: utf-8 -*-
"""
@author: Jean-Gabriel Simard
"""
from core.operations import Reduce_sum, Log, Absolute

#for classification
def cross_entropy(y_estimated, y):
    return -Reduce_sum(y * Log(y_estimated))

def cross_entropy2(y_estimated, y):
    return -Reduce_sum(y * Log(y_estimated) + (~y) * Log(~y_estimated))

#for regression
def l2(y_estimated, y):
    return Reduce_sum((y_estimated - y)**2)

def l1(y_estimated, y):
    return Reduce_sum(Absolute(y_estimated - y))