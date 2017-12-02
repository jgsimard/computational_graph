# -*- coding: utf-8 -*-
"""
@author: Jean-Gabriel Simard
"""

from graph import Operation, Parameter
from queue import Queue
import numpy as np

def compute_gradients(loss):
    # grad_table[node] will contain the gradient of the loss w.r.t. the node's output
    grad_table = {}

    # The gradient of the loss with respect to the loss is just 1
    grad_table[loss] = 1

    # breadth-first search, backwards from the loss
    visited = set()
    queue = Queue()
    visited.add(loss)
    queue.put(loss)

    while not queue.empty():
        node = queue.get()

        if node != loss:
            # Compute the gradient of the loss with respect to this node's output
            grad_table[node] = 0

            for consumer in node.consumers:   
                loss_grads_wrt_consumer_inputs = consumer.gradient(grad_table[consumer])
                
                if len(consumer.input_nodes) == 1: 
                    # If 1 input node, lossgrads_wrt_consumer_inputs == scalar
                    grad_table[node] += loss_grads_wrt_consumer_inputs

                else:
                    # lossgrads_wrt_consumer_inputs == array 
                    # Get the gradient of the loss with respect to node
                    grad_table[node] += loss_grads_wrt_consumer_inputs[consumer.input_nodes.index(node)]

        # Append each input node to the queue
        if isinstance(node, Operation):
            for input_node in node.input_nodes:
                if not input_node in visited:
                    visited.add(input_node)
                    queue.put(input_node)

    # Return gradients for each visited node
    return grad_table

class GradientDescent(Operation):
    
    def __init__(self, loss, learning_rate = None):
        super().__init__()
        self.loss = loss
        if learning_rate is None:
            learning_rate = 0.1
        self.learning_rate = learning_rate
                

    def compute(self):
        grad_table = compute_gradients(self.loss)
        for node in grad_table:
            if type(node) == Parameter:
                grad = grad_table[node]
                node.value -= self.learning_rate * np.mean(grad,axis = 0)
                

class Momentum(Operation):
    
    def __init__(self, loss, learning_rate = None, gamma = None):
        super().__init__()
        self.loss = loss
        
        if learning_rate is None:
            learning_rate = 0.1
        self.learning_rate = learning_rate
        
        if gamma is None:
            gamma = 0.5
        self.gamma = gamma
        
    def compute(self):
        grad_table = compute_gradients(self.loss)
              
        i = 0
        for node in grad_table:
            if type(node) == Parameter:
                # Retrieve gradient for this variable
                grad = grad_table[node]
                print(i)
                i+=1
                print(grad)
                print(np.mean(grad,axis = 0))
                node.value -= self.learning_rate * np.mean(grad,axis = 0)