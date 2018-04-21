# -*- coding: utf-8 -*-
"""
@author: Jean-Gabriel Simard
"""
from core.operations import Reduce_sum, Log, Absolute
from core.graph import scope

#for classification
def cross_entropy(y_estimated, y):
    with scope('Loss : Cross_entropy', y_estimated.graph):
        return -Reduce_sum(y * Log(y_estimated))

def cross_entropy2(y_estimated, y):
    with scope('Loss : Cross_entropy', y_estimated.graph):
        return -Reduce_sum(y * Log(y_estimated) + (~y) * Log(~y_estimated))

#for regression
def l2(y_estimated, y):
    with scope('Loss : L2', y_estimated.graph):
        return Reduce_sum((y_estimated - y)**2)

def l1(y_estimated, y):
    with scope('Loss : L1', y_estimated.graph):
        return Reduce_sum(Absolute(y_estimated - y))