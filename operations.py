# -*- coding: utf-8 -*-
"""
@author: Jean-Gabriel Simard

Contains operations that can be used to build a network
"""

import numpy as np
from graph import Operation
from scipy.special import expit
    
    
class add(Operation):
    """Returns x + y element-wise.
    """
    def __init__(self, x, y):
        super().__init__([x, y])

    def compute(self, x_value, y_value):
        return x_value + y_value
    
    def gradient(self,  grad):        
        def add_grad_wrt(a, grad):
            grad_wrt_a = grad
            while np.ndim(grad_wrt_a) > len(a.shape):
                grad_wrt_a = np.sum(grad_wrt_a, axis=0)
            for axis, size in enumerate(a.shape):
                if size == 1:
                    grad_wrt_a = np.sum(grad_wrt_a, axis=axis, keepdims=True)
            return grad_wrt_a
                
        return [add_grad_wrt(self.inputs[0], grad), add_grad_wrt(self.inputs[1], grad)]

class substract(add):
    """Returns x - y element-wise.
    """
    def __init__(self, x, y):
        yp = negative(y)
        super(add,self).__init__([x, yp])

class negative(Operation):
    """Computes the negative of x element-wise.
    """

    def __init__(self, x):
        super().__init__([x])

    def compute(self, x_value):
        return -x_value
    
    def gradient(self, grad):
        return -grad
       
class inverse(Operation):
    """Returns 1-x element-wise.
    """
    def __init__(self, x):
        super().__init__([x])
    
    def compute(self, x_value):
        return 1 - x_value
    
    def gradient(self, grad):
        return -grad
    
class absolute(Operation):
    """Computes the absolute value of x element-wise.
    """

    def __init__(self, x):
        super().__init__([x])

    def compute(self, x_value):
        return np.absolute(x_value)
    
    def gradient(self, grad):
        return np.sign(self.inputs)*grad
       
class matmul(Operation):
    """Multiplies matrix a by matrix b, producing a * b.
    """

    def __init__(self, a, b):
        super().__init__([a, b])

    def compute(self, a_value, b_value):
        return a_value.dot(b_value)
    
    def gradient(self, grad):
        return [grad.dot(self.inputs[1].T), self.inputs[0].T.dot(grad)]
    
class sigmoid(Operation):
    """Returns the sigmoid(1 / (1 + exp(-x))  of x element-wise.
    """

    def __init__(self, a):
        super().__init__([a])

    def compute(self, a_value):
        return expit(np.clip( a_value, -5000, 5000 )) #clip to get rid of nan values
    
    def gradient(self, grad):
        return grad * self.output * (1 - self.output)

class softmax(Operation):
    """Returns the softmax of a.
    """

    def __init__(self, a):
        super().__init__([a])

    def compute(self, a_value):
        return np.exp(a_value) / np.sum(np.exp(a_value), axis=-1)[:, None]
    
    def gradient(self, grad):
        return (grad - np.reshape(np.sum(grad * self.output, -1), [-1, 1] )) * self.output   
    
class log(Operation):
    """Computes the natural logarithm of x element-wise.
    """
    
    def __init__(self, x):
        super().__init__([x])
    
    def compute(self, x_value):
#        return np.log(np.clip(x_value,10**-5,None))
        return np.log(x_value)
    
    def gradient(self, grad):
        return grad/(self.inputs[0])


class multiply(Operation):
    """Returns x * y element-wise.
    """

    def __init__(self, x, y):
        super().__init__([x, y])

    def compute(self, x_value, y_value):
        return x_value * y_value
    
    def gradient(self, grad):
        return [grad * self.inputs[1], grad * self.inputs[0]]
 
    
class reduce_sum(Operation):
    """Computes the sum of elements across dimensions of a tensor.
    """

    def __init__(self, A, axis=None):
        super().__init__([A])
        self.axis = axis

    def compute(self, A_value):
        return np.sum(A_value, self.axis)
    
    def gradient(self, grad):
        
        A = self.inputs[0]       
        output_shape = np.array(A.shape)
        output_shape[self.axis] = 1
        tile_scaling = A.shape // output_shape
        grad = np.reshape(grad, output_shape)
        return np.tile(grad, tile_scaling)


class square(Operation):
    """Computes x**2 element-wise.
    """

    def __init__(self, x):
        super().__init__([x])

    def compute(self, x_value):
        return np.square(x_value)
    
    def gradient(self, grad):
        return 2 * self.inputs[0] * grad