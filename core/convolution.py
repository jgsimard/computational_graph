# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 12:06:02 2018

@author: Jean-Gabriel Simard
"""
from core.graph import Operation
import numpy as np
class Convolution2DNaive(Operation):
    '''
    2D convolution because the kernel only moves along 2 dimensions of the 3d 
    individual input
    X: Ni x C x H  x W
    W: FN x C x FH x FW
    b: F  x 1
    
    Ni number of input
    C  number of image channel
    H  height of image
    W  width of the image
    
    FN number of filter in the filter map W
    FH height of the filter, and finally
    FW width of the filter
    '''
    
    def __init__(self, x, w, b, stride = 1, padding = 1):
        super().__init__([x, w, b])
        self.name = 'Convolution2DNaive'
        self.stride = stride
        self.pad = padding
        self.x_shape = None
        
        
    def compute(self, x_value, w_value, b_value):
        self.x_shape = x_value.shape
        n_input,  x_c, x_h, x_w = x_value.shape
        f_n, f_c, f_h, f_w = w_value.shape
        
        out_h = (x_h - f_h + 2 * self.pad) / self.S + 1
        out_w = (x_w - f_w + 2 * self.pad) / self.S + 1
        
        assert out_h % 1.0 == 0
        assert out_w % 1.0 == 0
        assert x_c == f_c
        
        out_h, out_w = int(out_h), int(out_w)
        
        out = np.zeros((n_input, f_n, out_h, out_w))
        
        x_value_padded = np.pad(x_value, 
                                ((0,0),(0,0),(self.pad, self.pad),(self.pad, self.pad)),
                                'constant', constant_values=(0))
        
        for i in range(n_input): #item in batch
            for f in range(f_n): # item channel
                for h in range(0,x_h, self.stride):
                    for w in range(0,x_w, self.stride):
                        out[i,f,h/self.stride, w/self.stride] = ...
                        np.sum(x_value_padded[i, :, h : h+f_h, w : w+f_w] * w_value[f,:,:,:]) + b_value[f]
                    
        return out

    def gradient(self, grad):
        
        n_input,  x_c, x_h, x_w = self.input_nodes[0].output.shape
        f_n,      f_c, f_h, f_w = self.input_nodes[1].output.shape
        N, F, out_h, out_w = grad.shape
                
        x_value_padded = np.pad(self.input_nodes[0].output, 
                                ((0,0),(0,0),(self.pad, self.pad),(self.pad, self.pad)),
                                'constant', constant_values=(0))
        # dx
        dx = np.zeros(x_value_padded)
        for i in range(n_input): #item in batch
            for f in range(f_n): # item channel
                for h in range(0,x_h, self.stride):
                    for w in range(0,x_w, self.stride):
                        dx[i,:,x_h:x_h+f_h, x_w:x_w+f_w] += grad[i,f,x_h/self.stride, x_w/self.stride] * self.input_nodes[1][f,:,:,:]
                        
        dx = np.delete(dx, range(self.pad) + range(x_h + self.pad, x_h + 2 * self.pad, 1), axis = 2) # delete excess rows
        dx = np.delete(dx, range(self.pad) + range(x_w + self.pad, x_w + 2 * self.pad, 1), axis = 3) # delete excess cols
        
        #dw
        dw = np.zeros(self.input_nodes[1].output.shape)
        for i in range(n_input): #item in batch
            for f in range(f_n): # item channel
                for h in range(0, out_h):
                    for w in range(0, out_w):
                        dw[f,:,:,:] += grad[i,f,h,w] * x_value_padded[i, :, h*self.stride:h*self.stride + f_h, w*self.stride:w*self.stride + f_w]
                        
        #db
        db = np.zeros(self.input_nodes[2].output.shape)
        for f in range(f_n):
            db[f] = np.sum(grad[:,f,:,:])
        
        return [dx, dw, db]
        

class im2col (Operation):
    
    def __init__(self, x), stride = 1, padding = 1:
        super().__init__([x])
        self.name = "im2col"
    
    def compute (self, x_value):
        n_input,  x_c, x_h, x_w = x_value.shape
        f_n, f_c, f_h, f_w = w_value.shape
    
    def gradient(self, grad):
        pass

class Convolution2D(Operation):
        pass

class Convolution(Operation):
    
    def __init__(self,x, W, b, stride = 1, padding = 1):
        super().__init__([x,W,b])
        self.name = 'Convolution 2D'
        self.w_shape = W.value.shape
        self.b_shape = b.value.shape
        self.stride = stride
        self.pad = padding
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
#      print('cols', cols.shape)
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
        dW = dW.reshape(self.input_nodes[1].output.shape)
    
        W_reshape = self.input_nodes[1].output.reshape(n_filter, -1)
        dX_col = W_reshape.T @ grad_reshaped
        dX = self.col2im_indices(dX_col, self.input_nodes[0].output.shape, self.FH, self.FW, padding=self.pad, stride=self.stride)
    
        return [dX, dW, db]