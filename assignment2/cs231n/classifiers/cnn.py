import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    C,H,W = input_dim
    std = weight_scale
    self.params['W1'] = std * np.random.randn(num_filters, C, filter_size, filter_size)
    self.params['b1'] = np.zeros(num_filters)
    self.params['W2'] = std * np.random.randn((H/2)*(W/2)*num_filters, hidden_dim)
    self.params['b2'] = np.zeros(hidden_dim)
    self.params['W3'] = std * np.random.randn(hidden_dim, num_classes)
    self.params['b3'] = np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    cache = {}
    flow, cache[1] = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    flow, cache[2] = affine_relu_forward(flow, W2, b2)
    flow, cache[3] = affine_forward(flow, W3, b3)
    scores = flow
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, dscore = softmax_loss(scores, y)
    dout2, grads['W3'], grads['b3'] = affine_backward(dscore, cache[3])
    dout1, grads['W2'], grads['b2'] = affine_relu_backward(dout2, cache[2])
    dout0, grads['W1'], grads['b1'] = conv_relu_pool_backward(dout1, cache[1])

    # regularization
    loss += 0.5 * self.reg * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2))
    grads['W1'] += self.reg * W1
    grads['W2'] += self.reg * W2
    grads['W3'] += self.reg * W3
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads

  
class MyConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  cbr:conv-batch normalization-relu
  cbr1 - pool1 - cbr2_1 - cbr2_2 - pool2 - cbr3_1 - cbr3_2 - pool3 - fc4 - softmax5

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  def __init__(self, input_dim=(3, 32, 32), num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    C,H,W = input_dim
    std = weight_scale

    def init_cbr_layer(depth_in, depth_out, idx_str):
      self.params['Wconv'+idx_str] = std * np.random.randn(depth_out, depth_in, 3, 3)
      self.params['bconv'+idx_str] = np.zeros(depth_out)
      self.params['gamma_conv'+idx_str] = np.ones(depth_out)
      self.params['beta_conv'+idx_str] = np.zeros(depth_out)

    init_cbr_layer(C, 16, '1')
    init_cbr_layer(16, 16, '2_1')
    init_cbr_layer(16, 32, '2_2')
    init_cbr_layer(32, 32, '3_1')
    init_cbr_layer(32, 64, '3_2')

    hidden_dim = 512
    self.params['Wfc4'] = std * np.random.randn((H/2/2/2)*(W/2/2/2)*64, hidden_dim)
    self.params['bfc4'] = np.zeros(hidden_dim)
    self.params['gamma_fc4'] = np.ones(hidden_dim)
    self.params['beta_fc4'] = np.zeros(hidden_dim)
    self.params['Wfc5'] = std * np.random.randn(hidden_dim, num_classes)
    self.params['bfc5'] = np.zeros(num_classes)

    self.bn_params = {'conv1':{'mode': 'train'},
                      'conv2_1':{'mode': 'train'},
                      'conv2_2':{'mode': 'train'},
                      'conv3_1':{'mode': 'train'},
                      'conv3_2':{'mode': 'train'},
                      'fc4':{'mode': 'train'},
                      }
    self.dropout_param = {'mode': 'train', 'p': 0.5}

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    
    conv_param = {'stride': 1, 'pad': 1}
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
    mode = 'test' if y is None else 'train'
    self.dropout_param['mode'] = mode
    for _,v in self.bn_params.iteritems():
      v['mode'] = mode

    scores = None
    cache = {}
    def build_cbr_layer(input_data, idx_str):
      cbr, cache['cbr'+idx_str] = conv_bn_relu_forward(input_data, 
                self.params['Wconv'+idx_str], self.params['bconv'+idx_str], conv_param,
                self.params['gamma_conv'+idx_str], self.params['beta_conv'+idx_str], self.bn_params['conv'+idx_str])
      return cbr

    cbr1 = build_cbr_layer(X, '1')
    pool1, cache['pool1'] = max_pool_forward_fast(cbr1, pool_param)
    cbr2_1 = build_cbr_layer(pool1, '2_1')
    cbr2_2 = build_cbr_layer(cbr2_1, '2_2')
    pool2, cache['pool2'] = max_pool_forward_fast(cbr2_2, pool_param)
    cbr3_1 = build_cbr_layer(pool2, '3_1')
    cbr3_2 = build_cbr_layer(cbr3_1, '3_2')
    pool3, cache['pool3'] = max_pool_forward_fast(cbr3_2, pool_param)
    fc4,cache['fc4'] = affine_bn_relu_forward(pool3, self.params['Wfc4'], self.params['bfc4'],
                self.params['gamma_fc4'], self.params['beta_fc4'], self.bn_params['fc4'])
    fc4_dp, cache['dropout_fc4'] = dropout_forward(fc4, self.dropout_param)
    fc5,cache['fc5'] = affine_forward(fc4_dp, self.params['Wfc5'], self.params['bfc5'])
    scores = fc5

    if y is None:
      return scores
    
    loss, grads = 0, {}
    loss, dscore = softmax_loss(scores, y)
    dfc4_dp, grads['Wfc5'], grads['bfc5'] = affine_backward(dscore, cache['fc5'])
    dfc4 = dropout_backward(dfc4_dp, cache['dropout_fc4'])
    dpool3, grads['Wfc4'], grads['bfc4'], grads['gamma_fc4'], grads['beta_fc4'] = affine_bn_relu_backward(dfc4, cache['fc4'])
    dcbr3_2 = max_pool_backward_fast(dpool3, cache['pool3'])
    dcbr3_1, grads['Wconv3_2'], grads['bconv3_2'], grads['gamma_conv3_2'], grads['beta_conv3_2'] = conv_bn_relu_backward(dcbr3_2, cache['cbr3_2'])
    dpool2 , grads['Wconv3_1'], grads['bconv3_1'], grads['gamma_conv3_1'], grads['beta_conv3_1'] = conv_bn_relu_backward(dcbr3_1, cache['cbr3_1'])
    dcbr2_2 = max_pool_backward_fast(dpool2, cache['pool2'])
    dcbr2_1, grads['Wconv2_2'], grads['bconv2_2'], grads['gamma_conv2_2'], grads['beta_conv2_2'] = conv_bn_relu_backward(dcbr2_2, cache['cbr2_2'])
    dpool1 , grads['Wconv2_1'], grads['bconv2_1'], grads['gamma_conv2_1'], grads['beta_conv2_1'] = conv_bn_relu_backward(dcbr2_1, cache['cbr2_1'])
    dcbr1 = max_pool_backward_fast(dpool1, cache['pool1'])
    dinput, grads['Wconv1'], grads['bconv1'], grads['gamma_conv1'], grads['beta_conv1'] = conv_bn_relu_backward(dcbr1, cache['cbr1'])

    # regularization
    for s in ['Wconv1','Wconv2_1','Wconv2_2','Wconv3_1','Wconv3_2','Wfc4','Wfc5']:
      loss += 0.5 * self.reg * np.sum(self.params[s]**2)
      grads[s] += self.reg * self.params[s]
    
    return loss, grads
  
