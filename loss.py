# -*- coding: utf-8 -*-
"""
@author: Jean-Gabriel Simard
"""
#import operations as op
from operations import negative, reduce_sum, multiply, log, square, substract, inverse, add, absolute

#for classification
def cross_entropy(y_a, y):
    return negative(reduce_sum(multiply(y, log(y_a))))

def cross_entropy2(y_a, y):
    return negative(reduce_sum(add(multiply(y, log(y_a)), multiply(inverse(y), log(inverse(y_a))))))

#for regression
def l2(y_a, y):
    return reduce_sum(square(substract(y_a, y)))


def l0(y_a, y):
    return reduce_sum(absolute(substract(y_a, y)))