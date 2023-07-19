import numpy as np
from nndl.layers import *
import pdb


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  pad = conv_param['pad']
  stride = conv_param['stride']

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the forward pass of a convolutional neural network.
  #   Store the output as 'out'.
  #   Hint: to pad the array, you can use the function np.pad.
  # ================================================================ #
  N, C, H, W = x.shape  
  F, C, HH, WW = w.shape

  x_pad = (np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant'))
  out_height = np.int(((H + 2 * pad - HH) / stride) + 1) 
  out_width = np.int(((W + 2 * pad - WW) / stride) + 1) 
  out = np.zeros([N, F, out_height, out_width])

  for img in range(N):  
    for kernal in range(F):  
      for row in range(out_height): 
        for col in range(out_width):
          out[img, kernal, row, col] = np.sum(w[kernal,:,:,:] * x_pad[img, :, row*stride:row*stride+HH, col*stride:col*stride+WW]) + b[kernal]

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #
    
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None

  N, F, out_height, out_width = dout.shape
  x, w, b, conv_param = cache
  
  stride, pad = [conv_param['stride'], conv_param['pad']]
  xpad = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), mode='constant')
  num_filts, _, f_height, f_width = w.shape

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the backward pass of a convolutional neural network.
  #   Calculate the gradients: dx, dw, and db.
  # ================================================================ #
  dx = np.zeros_like(x)                                # create dx, should have the same shape as x: (N, C, H, W)
  dw = np.zeros_like(w)                                # create dw, should have the same shape as w: (F, C, HH, WW)
  dxpad = np.zeros_like(xpad)                             # create dxpad, should have the same shape as xpad 

  db = dout.sum(axis=(0,2,3))

  for img in range(N): 
    for kernal in range(F):  
      for row in range(out_height): 
        for col in range(out_width): 
            dw[kernal,:,:,:] += dout[img, kernal, row, col] * xpad[img, :, row*stride:row*stride+f_height, col*stride:col*stride+f_width]
            dxpad[img,:, row*stride:row*stride+f_height, col*stride:col*stride+f_width] += dout[img, kernal, row,col] * w[kernal,:,:,:]
            
  dx = dxpad[:,:,pad:-pad,pad:-pad] 

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #

  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  
  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling forward pass.
  # ================================================================ #

  h_pool,w_pool,stride = pool_param['pool_height'],pool_param['pool_width'],pool_param['stride']

  N, F, H, W = x.shape
  h_out = int((H - h_pool)/stride + 1)
  w_out = int((W - w_pool)/stride + 1)
  out = np.zeros((N, F, h_out, w_out))

  for h in np.arange(h_out):
    for w in np.arange(w_out):
        hh = h*stride
        ww = w*stride
        out[:,:,h,w] = np.amax(x[:,:,hh:hh + h_pool,ww:ww + w_pool],axis=(2,3))    
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 
  cache = (x, pool_param)
  return out, cache

def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  x, pool_param = cache
  pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling backward pass.
  # ================================================================ #
    
  N, F, H, W = x.shape
  h_out = int(1 + (H - pool_height) / stride)
  w_out = int(1 + (W - pool_width) / stride)
  dx = np.zeros((N, F, H, W))


  for n in range(N):
      for f in range(F):
          for h in range(h_out):
              for w in range(w_out):
                   hh = h * stride
                   ww = w * stride
                   x_pool = x[n, f, hh:hh + pool_height, ww:ww + pool_width]
                   mask = (x_pool==np.amax(x_pool))
                   dx[n, f, hh:hh + pool_height, ww:ww + pool_width] += dout[n, f, h, w] * mask
# reference:https://www.kaggle.com/code/valentynsichkar/convolutional-neural-network-from-scratch-cifar10/notebook
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return dx

def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm forward pass.
  #
  #   You may find it useful to use the batchnorm forward pass you 
  #   implemented in HW #4.
  # ================================================================ #
  N, C, H, W = x.shape
  x = np.swapaxes(x,0,1)
  x = x.reshape(C,N*H*W).T
  out, cache = batchnorm_forward(x, gamma, beta, bn_param)
  out = out.T.reshape(C,N,H,W).swapaxes(0,1)
  
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm backward pass.
  #
  #   You may find it useful to use the batchnorm forward pass you 
  #   implemented in HW #4.
  # ================================================================ #
  N, C, H, W = dout.shape
  dout = dout.swapaxes(0,1).reshape(C,N*H*W).T
  dx, dgamma, dbeta = batchnorm_backward(dout, cache)
  dx = dx.T.reshape(C,N,H,W).swapaxes(0,1)

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return dx, dgamma, dbeta