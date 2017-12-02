# -*- coding: utf-8 -*-
"""
@author: Jean-Gabriel Simard
"""
import operations as op

#for classification
def cross_entropy(y_a, y):
    return op.negative(op.reduce_sum(op.reduce_sum(op.multiply(y, op.log(y_a)))))

#for regression
def l2(y_a, y):
    return op.reduce_sum(op.reduce_sum(op.square(op.substract(y_a, y))))