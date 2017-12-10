# -*- coding: utf-8 -*-
"""
@author: Jean-Gabriel Simard
"""
#import operations as op
from operations import reduce_sum, log, absolute

#for classification
def cross_entropy(y_estimated, y):
    return -reduce_sum(y * log(y_estimated))

def cross_entropy2(y_estimated, y):
    return -reduce_sum(y * log(y_estimated) + (~y) * log(~y_estimated))

#for regression
def l2(y_estimated, y):
    return reduce_sum((y_estimated - y)**2)

def l1(y_estimated, y):
    return reduce_sum(absolute(y_estimated - y))