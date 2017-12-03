# -*- coding: utf-8 -*-
"""
@author: Jean-Gabriel Simard
"""
#import operations as op
from operations import negative, reduce_sum, multiply, log, square, substract, inverse, add

#for classification
def cross_entropy(y_a, y):
    return negative(reduce_sum(reduce_sum(multiply(y, log(y_a)))))

def cross_entropy2(y_a, y):
    return add(cross_entropy(y_a, y), \
               cross_entropy(inverse(y_a), inverse(y)))
    

#for regression
def l2(y_a, y):
    return reduce_sum(reduce_sum(square(substract(y_a, y))))