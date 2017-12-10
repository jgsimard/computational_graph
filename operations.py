# -*- coding: utf-8 -*-
"""
@author: Jean-Gabriel Simard

Contains operations that can be used to build a network
"""

import numpy as np
from graph import Operation, Node
from scipy.special import expit
    
class add(Operation):
    """Computes x + y element-wise.
    """
    def __init__(self, x, y):
        super().__init__([x, y])
        self.name = 'Add'

    def compute(self, x_value, y_value):
        return x_value + y_value
    
    def gradient(self,  grad):
        return [grad, grad]
    
class substract(add):
    """Returns x - y element-wise.
    """
    def __init__(self, x, y):
        super().__init__(x, negative(y))
        self.name = 'Substract'

class negative(Operation):
    """Computes the negative of x element-wise.
    """

    def __init__(self, x):
        super().__init__([x])
        self.name = 'Negative'

    def compute(self, x_value):
        return -x_value
    
    def gradient(self, grad):
        return -grad
       
class inverse(Operation):
    """Returns 1-x element-wise.
    """
    def __init__(self, x):
        super().__init__([x])
        self.name = 'inverse'
    
    def compute(self, x_value):
        return 1 - x_value
    
    def gradient(self, grad):
        return -grad
    
class absolute(Operation):
    """Computes the absolute value of x element-wise.
    """

    def __init__(self, x):
        super().__init__([x])
        self.name = 'Absolute'

    def compute(self, x_value):
        return np.absolute(x_value)
    
    def gradient(self, grad):
        return np.sign(self.inputs)*grad
       
class matmul(Operation):
    """Multiplies matrix a by matrix b, producing a * b.
    """

    def __init__(self, a, b):
        super().__init__([a, b])
        self.name = 'Matrix multiplication'

    def compute(self, a_value, b_value):
        return a_value.dot(b_value)
    
    def gradient(self, grad):
        return [grad.dot(self.inputs[1].T), self.inputs[0].T.dot(grad)]
    
class sigmoid(Operation):
    """Returns the sigmoid(1 / (1 + exp(-x))  of x element-wise.
    """

    def __init__(self, a):
        super().__init__([a])
        self.name = 'Sigmoid'

    def compute(self, a_value):
        return expit(np.clip( a_value, -5000, 5000 )) #clip to get rid of nan values
    
    def gradient(self, grad):
        return grad * self.output * (1 - self.output)

class softmax(Operation):
    """Returns the softmax of a.
    """

    def __init__(self, a):
        super().__init__([a])
        self.name = 'Softmax'

    def compute(self, a_value):
        return np.exp(a_value) / np.expand_dims(np.exp(a_value).sum(axis=-1), axis = -1)
    
    def gradient(self, grad):
        return (grad - np.expand_dims(np.sum(grad * self.output, -1), axis = -1 )) * self.output  
    
class log(Operation):
    """Computes the natural logarithm of x element-wise.
    """
    
    def __init__(self, x):
        super().__init__([x])
        self.name = 'Ln'
    
    def compute(self, x_value):
        return np.log(x_value)
    
    def gradient(self, grad):
        return grad/self.inputs[0]


class multiply(Operation):
    """Returns x * y element-wise.
    """

    def __init__(self, x, y):
        super().__init__([x, y])
        self.name = 'Element wise multiplication'

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
        self.name = 'Reduce sum'

    def compute(self, A_value):
        return np.sum(A_value, self.axis)
    
    def gradient(self, grad):
        
        A = self.inputs[0]       
        output_shape = np.array(A.shape)
        output_shape[self.axis] = 1
        tile_scaling = A.shape // output_shape
        grad = np.reshape(grad, output_shape)
        return np.tile(grad, tile_scaling)

class pow_op(Operation):
    def __init__(self, x, y):
        super().__init__([x])
        self.exponent = y
        self.name = 'Power ' + str(y)

    def compute(self, x_value):
        return np.power(x_value, self.exponent)
    
    def gradient(self, grad):
        return self.exponent * self.inputs[0] * grad
    
class square(pow_op):
    """Computes x**2 element-wise.
    """
    def __init__(self, x):
        super().__init__(x,2.0)
        self.name = 'Square'
        
class leaky_relu(Operation):
    """Computes max(ax,x) element-wise. element-wise.
    """

    def __init__(self, x, alpha = 0.1):
        super().__init__([x])
        self.alpha = alpha
        self.name = 'Leaky Relu (' + str(alpha) + ')'

    def compute(self, x_value):
        return np.maximum(x_value * self.alpha, x_value)
    
    def gradient(self, grad):
        filt = self.output
        filt[filt >= 0] = 1.0
        filt[filt < 0]  = self.alpha
        return filt * grad

class relu(leaky_relu):
    """Computes max(0,x) element-wise.
    """
    def __init__(self, x):
        super().__init__(x,0.0)
        self.name = 'Relu'
        

class Convolution(Operation):
    
    '''
    X: D  x C x H  x W
    W: NF x C x HF x HW
    b: F  x 1
    
    D  number of input
    C  number of image channel
    H  height of image
    W  width of the image
    NF number of filter in the filter map W
    HF height of the filter, and finally
    HW width of the filter
    
    '''
    def __init__(self,x, W, b, stride = 1, pad = 1):
        super().__init__([x,W,b])
        self.name = 'Convolution 2D'
        self.w_shape = W.value.shape
        self.b_shape = b.value.shape
        self.stride = stride
        self.pad = pad
        self.FH = self.w_shape[2]
        self.FW = self.w_shape[3]
        
    def get_im2col_indices(self, x_shape, field_height, field_width, padding=1, stride=1):
      # First figure out what the size of the output should be
      N, C, H, W = x_shape
      assert (H + 2 * padding - field_height) % stride == 0
      assert (W + 2 * padding - field_height) % stride == 0
      out_height = int((H + 2 * padding - field_height) / stride + 1)
      out_width = int((W + 2 * padding - field_width) / stride + 1)
          
      i0 = np.repeat(np.arange(field_height), field_width)
      i0 = np.tile(i0, C)
      i1 = stride * np.repeat(np.arange(out_height), out_width)
      j0 = np.tile(np.arange(field_width), field_height * C)
      j1 = stride * np.tile(np.arange(out_width), out_height)
      i = i0.reshape(-1, 1) + i1.reshape(1, -1)
      j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    
      k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)
    
      return (k, i, j)
    
    
    def im2col_indices(self, x, field_height, field_width, padding=1, stride=1):
      """ An implementation of im2col based on some fancy indexing """
      # Zero-pad the input
      p = padding
      x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
    
      k, i, j = self.get_im2col_indices(x.shape, field_height, field_width, padding,
                                   stride)
    
      cols = x_padded[:, k, i, j]
      C = x.shape[1]
      cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
      return cols
    
    
    def col2im_indices(self, cols, x_shape, field_height=3, field_width=3, padding=1,
                       stride=1):
      """ An implementation of col2im based on fancy indexing and np.add.at """
      N, C, H, W = x_shape
      H_padded, W_padded = H + 2 * padding, W + 2 * padding
      x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
      k, i, j = self.get_im2col_indices(x_shape, field_height, field_width, padding,
                                   stride)
      cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
      cols_reshaped = cols_reshaped.transpose(2, 0, 1)
      np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
      if padding == 0:
        return x_padded
      return x_padded[:, :, padding:-padding, padding:-padding]
        
    def compute(self, x_value, w_value, b_value):
        n_filters, d_filter, h_filter, w_filter = self.w_shape
        n_x, d_x, h_x, w_x = x_value.shape
        
        h_out = (h_x - h_filter + 2 * self.pad) / self.stride + 1
        w_out = (w_x - w_filter + 2 * self.pad) / self.stride + 1
        assert (h_out % 1.0 == 0)
        assert (w_out % 1.0 == 0)

        h_out, w_out = int(h_out), int(w_out)
        
        self.X_col = self.im2col_indices(x_value, h_filter, w_filter, padding=self.pad, stride=self.stride)
        self.W_col = w_value.reshape(n_filters, -1)
        
        out = self.W_col.dot(self.X_col) + b_value[:,None]
        out = out.reshape(n_filters, h_out, w_out, n_x)
        out = out.transpose(3, 0, 1, 2)
        
        return out

    def gradient(self, grad):
        n_filter, d_filter, h_filter, w_filter = self.w_shape
    
        db = np.sum(grad, axis=(0, 2, 3))
        db = db.reshape(n_filter, -1)
    
        grad_reshaped = grad.transpose(1, 2, 3, 0).reshape(n_filter, -1)
        dW = grad_reshaped.dot(self.X_col.T)
        dW = dW.reshape(self.inputs[1].shape)
    
        W_reshape = self.inputs[1].reshape(n_filter, -1)
        dX_col = W_reshape.T @ grad_reshaped
        dX = self.col2im_indices(dX_col, self.inputs[0].shape, self.FH, self.FW, padding=self.pad, stride=self.stride)
    
        return [dX, dW, db]

class maximum(Operation):
    def __init__(x, y):
        super().__init__([x, y])
    
    def compute(self, x_value, y_value):
        return np.maximum(x_value, y_value)
    
    def gradient(self, grad):
        dx = np.zeros(self.inputs[0].shape)
        dx[self.inputs[0]>self.inputs[1]] = 1.0
        dy = 1.0 - dx
        return[dx*grad, dy*grad]

class Flatten(Operation):
    def __init__(self,x):
        super().__init__([x])
        self.shape = 0
        self.name = "Flatten"
    
    def compute(self, x_value):
        self.shape = x_value.shape
        return x_value.reshape((x_value.shape[0],-1))
    
    def gradient(self, grad):
        return grad.reshape(self.shape)
        
    
class Max_pool(Operation):
    def __init__(x):
        super().__init__([x])
    
    def compute(self, x_value):
        pass
    
    def gradient(self, grad):
        pass
    
'''
Operation everloading to simplify the construction of a graph
'''
Node.__add__ = lambda x, y : add(x,y)
Node.__sub__ = lambda x, y : substract(x,y)
Node.__mul__ = lambda x, y : multiply(x,y)
Node.__pow__ = lambda x, y : pow_op(x,y)
Node.__neg__ = lambda x    : negative(x)
Node.__invert__ = lambda x : inverse(x)