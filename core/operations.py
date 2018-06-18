# -*- coding: utf-8 -*-
"""
@author: Jean-Gabriel Simard

Contains operations that can be used to build a network
"""

import numpy as np
from core.graph import Operation, Node
from scipy.special import expit
    
class Add(Operation):
    """Computes x + y element-wise.
    """
    def __init__(self, x, y):
        super().__init__([x, y])
        self.name = 'Add'

    def compute(self, x_value, y_value):
        return x_value + y_value
    
    def gradient(self,  grad):
        return [grad, grad]
    
class Negative(Operation):
    """Computes the negative of x element-wise.
    """

    def __init__(self, x):
        super().__init__([x])
        self.name = 'Negative'

    def compute(self, x_value):
        return -x_value
    
    def gradient(self, grad):
        return -grad
       
class Inverse(Operation):
    """Returns 1-x element-wise.
    """
    def __init__(self, x):
        super().__init__([x])
        self.name = 'Inverse'
    
    def compute(self, x_value):
        return 1 - x_value
    
    def gradient(self, grad):
        return -grad
    
class Absolute(Operation):
    """Computes the absolute value of x element-wise.
    """

    def __init__(self, x):
        super().__init__([x])
        self.name = 'Absolute'

    def compute(self, x_value):
        return np.absolute(x_value)
    
    def gradient(self, grad):
        return np.sign(self.input_nodes[0].output)*grad
       
class Matmul(Operation):
    """Multiplies matrix a by matrix b, producing a * b.
    """

    def __init__(self, a, b):
        super().__init__([a, b])
        self.name = 'Matrix multiplication'

    def compute(self, a_value, b_value):
        return a_value.dot(b_value)
    
    def gradient(self, grad):
        return [grad.dot(self.input_nodes[1].output.T), self.input_nodes[0].output.T.dot(grad)]
    
class Sigmoid(Operation):
    """Returns the sigmoid(1 / (1 + exp(-x))  of x element-wise.
    """

    def __init__(self, a):
        super().__init__([a])
        self.name = 'Sigmoid'

    def compute(self, a_value):
        return expit(np.clip( a_value, -5000, 5000 )) #clip to get rid of nan values
    
    def gradient(self, grad):
        return grad * self.output * (1 - self.output)

class Softmax(Operation):
    """Returns the softmax of a.
    """

    def __init__(self, a):
        super().__init__([a])
        self.name = 'Softmax'

    def compute(self, a_value):
        return np.exp(a_value) / np.expand_dims(np.exp(a_value).sum(axis=-1), axis = -1)
    
    def gradient(self, grad):
        return (grad - np.expand_dims(np.sum(grad * self.output, -1), axis = -1 )) * self.output  
    
class Log(Operation):
    """Computes the natural logarithm of x element-wise.
    """
    
    def __init__(self, x):
        super().__init__([x])
        self.name = 'Ln'
    
    def compute(self, x_value):
        return np.log(x_value)
    
    def gradient(self, grad):
        return grad/self.input_nodes[0].output


class Multiply(Operation):
    """Returns x * y element-wise.
    """

    def __init__(self, x, y):
        super().__init__([x, y])
        self.name = 'Element wise multiplication'

    def compute(self, x_value, y_value):
        return x_value * y_value
    
    def gradient(self, grad):
        return [grad * self.input_nodes[1].output, grad * self.input_nodes[0].output]
 
    
class Reduce_sum(Operation):
    """Computes the sum of elements across dimensions of a tensor.
    """

    def __init__(self, A, axis=None):
        super().__init__([A])
        self.axis = axis
        self.name = 'Reduce sum'

    def compute(self, A_value):
        return np.sum(A_value, self.axis)
    
    def gradient(self, grad):
        
        A = self.input_nodes[0].output       
        output_shape = np.array(A.shape)
        output_shape[self.axis] = 1
        tile_scaling = A.shape // output_shape
        grad = np.reshape(grad, output_shape)
        return np.tile(grad, tile_scaling)

class Pow_op(Operation):
    """Computes element wise exponentiation
    """
    def __init__(self, x, y):
        super().__init__([x])
        self.exponent = y
        self.name = 'Power ' + str(y)

    def compute(self, x_value):
        return np.power(x_value, self.exponent)
    
    def gradient(self, grad):
        return self.exponent * self.input_nodes[0].output * grad
    
class Square(Pow_op):
    """Computes x**2 element-wise.
    """
    def __init__(self, x):
        super().__init__(x,2.0)
        self.name = 'Square'
        
class Leaky_relu(Operation):
    """Computes max(ax,x) element-wise.
    """

    def __init__(self, x, alpha = 0.1):
        super().__init__([x])
        self.alpha = alpha
        self.name = 'Leaky Relu (' + str(alpha) + ')'

    def compute(self, x_value):
        return np.maximum(x_value * self.alpha, x_value)
    
    def gradient(self, grad):
        filt = self.output[self.output >= 0]
        filt[filt < 0]  = self.alpha
        return filt * grad

class Relu(Leaky_relu):
    """Computes max(0,x) element-wise.
    """
    def __init__(self, x):
        super().__init__(x,0.0)
        self.name = 'Relu'
    
class Maximum(Operation):
    def __init__(x, y):
        super().__init__([x, y])
    
    def compute(self, x_value, y_value):
        return np.maximum(x_value, y_value)
    
    def gradient(self, grad):
        dx = np.zeros(self.input_nodes[0].output.shape)
        dx[self.input_nodes[0].output>self.input_nodes[1].output] = 1.0
        dy = 1.0 - dx
        return[dx*grad, dy*grad]

class Flatten(Operation):
    def __init__(self,x):
        super().__init__([x])
        self.shape = None
        self.name = "Flatten"
    
    def compute(self, x_value):
        self.shape = x_value.shape
        return x_value.reshape((x_value.shape[0],-1))
    
    def gradient(self, grad):
        return grad.reshape(self.shape)
        
'''
Operation everloading to simplify the construction of a graph
'''
Node.__add__    = lambda x, y : Add(x,y)
Node.__sub__    = lambda x, y : Add(x,Negative(y))
Node.__mul__    = lambda x, y : Multiply(x,y)
Node.__pow__    = lambda x, y : Pow_op(x,y)
Node.__neg__    = lambda x    : Negative(x)
Node.__invert__ = lambda x    : Inverse(x)