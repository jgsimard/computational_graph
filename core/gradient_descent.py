# -*- coding: utf-8 -*-
"""
@author: Jean-Gabriel Simard
"""

from core.graph import Operation, Parameter, Placeholder
from queue import Queue
import numpy as np
def compute_gradients(loss):
    # grad_table[node] = gradient of the loss w.r.t. the node's output
    grad_table = {}
    grad_table[loss] = 1

    # breadth-first search, backwards from the loss
    visited = set()
    queue = Queue()
    visited.add(loss)
    queue.put(loss)

    while not queue.empty():
        node = queue.get()
        
        # Append each input node to the queue
        if isinstance(node, Operation):
            for input_node in node.input_nodes:
                if input_node not in visited:
                    visited.add(input_node)
                    queue.put(input_node)
        
        #compute grad of a node wrt to the loss
        if node != loss:
            grad_table[node] = 0
            
            if not isinstance(node, Placeholder): 
                for consumer in node.consumers: 
                    if consumer in grad_table:
                        loss_grads_wrt_consumer_inputs = consumer.gradient(grad_table[consumer])
                        if len(consumer.input_nodes) == 1: 
                            grad_table[node] += loss_grads_wrt_consumer_inputs
                        else:
                            grad_table[node] += loss_grads_wrt_consumer_inputs[consumer.input_nodes.index(node)]

    # Return gradients for each visited node
    return grad_table

class Gradient_Descent(Operation):
    
    def __init__(self, loss, learning_rate = None, decay = None ):
        super().__init__()
        self.loss = loss
        
        if learning_rate is None:
            learning_rate = 0.001
        self.learning_rate = learning_rate
        
        if decay is None:
            decay = 0.9999
        self.decay = decay

class Vanilla(Gradient_Descent):
    
    def __init__(self, loss, learning_rate = None):
        super().__init__(loss, learning_rate)
        self.name = 'Vanilla gradient descent'

    def compute(self):
        grad_table = compute_gradients(self.loss)
        for node in grad_table:
            if type(node) == Parameter:
                node.value -= self.learning_rate * np.mean(grad_table[node],axis = 0)
        self.learning_rate *= self.decay
                
class Momentum(Gradient_Descent):
    
    def __init__(self, loss, learning_rate = None, gamma = None):
        super().__init__(loss, learning_rate)
        self.name = 'Momentum gradient descent'
        
        if gamma is None:
            gamma = 0.5
        self.gamma = gamma
        
        self.past_grad = {}
        
    def compute(self):
        grad_table = compute_gradients(self.loss)
              
        for node in grad_table:
            if type(node) == Parameter:
                if node in self.past_grad:
                    self.past_grad[node] *= self.gamma
                    self.past_grad[node] += self.learning_rate * np.mean(grad_table[node], axis = 0)
                else:
                    self.past_grad[node]  = self.learning_rate * np.mean(grad_table[node], axis = 0)
                node.value -= self.past_grad[node]
        self.learning_rate *= self.decay
                
class Nesterov(Gradient_Descent):
    
    def __init__(self, loss, learning_rate = None, gamma = None):
        super().__init__(loss, learning_rate)
        self.name = 'Nesterov gradient descent'
        
        if gamma is None:
            gamma = 0.5
        self.gamma = gamma
        
        self.past_grad = {}
        
    def compute(self):
        if self.past_grad == {}:
            grad_table = compute_gradients(self.loss)
            for node in grad_table:
                if type(node) == Parameter:
                    self.past_grad[node] = self.learning_rate *  np.mean(grad_table[node], axis = 0)
                    node.value -= self.past_grad[node]   
        else:
            for node in self.past_grad:
                if type(node) == Parameter:
                    node.value -= self.gamma * self.past_grad[node] 
            grad_table = compute_gradients(self.loss)
            
            for node in grad_table:
                if type(node) == Parameter:
                    self.past_grad[node] *= self.gamma
                    self.past_grad[node] += self.learning_rate * np.mean(grad_table[node], axis = 0)
                    node.value -= self.past_grad[node]
 
class Adagrad(Gradient_Descent):
    
    def __init__(self, loss, learning_rate = None, gamma = None):
        super().__init__(loss, learning_rate)
        self.name = 'Adagrad gradient descent'
        
        self.past_grad = {}
        
    def compute(self):
        if self.past_grad == {}:
            grad_table = compute_gradients(self.loss)
            for node in grad_table:
                if type(node) == Parameter:
                    self.past_grad[node] = np.square(np.mean(grad_table[node], axis = 0))
#                    node.value -= self.past_grad[node]   
        else:
            grad_table = compute_gradients(self.loss)
            
            for node in grad_table:
                if type(node) == Parameter:
                    node.value -= self.learning_rate + (1/(np.sqrt(self.past_grad[node]) + 10**-8)) * np.mean(grad_table[node], axis = 0)
                    self.past_grad[node] += np.square(np.mean(grad_table[node], axis = 0))