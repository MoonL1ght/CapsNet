import numpy as np
import tensorflow as tf


'''
Constants.
'''

'''Small constant to prevent zero-math errors.'''
ERROR_CONST = 1e-9

'''Standart deviation.'''
STDDEV = 0.1

DTYPE_F = tf.float32
DTYPE_INT = tf.int32


'''
Capsule net class.
'''
class CapsNet(object):
  def __init__(self,
               input_dim,
               classes,
               conv_filters=256,
               conv_kernel_size=9,
               conv_stride=1,
               conv_padding='valid',
               conv_activaion=tf.nn.relu,
               pc_vec_dim=8,
               pc_capsules=32,
               pc_kernel_size=9,
               pc_stride=2,
               pc_padding='valid',
               pc_activaion=tf.nn.relu,
               routing_rounds=3,
               dc_vec_dim=16,
               alpha=0.5,
               m_plus=0.9,
               m_minus=0.1,
               decoder_layers=(512, 1024),
               decoder_loss_scale=0.0005,
               dtype_float=DTYPE_F,
               dtype_int=DTYPE_INT):
    '''
    Initializing CapsNet.
    Args:
      input_dim: Size of input data, tuple (x_dim, y_dim, filters).
      classes: Total number of classes.
      conv_filters: Number of filter in conv layer.
      conv_kernel_size: Conv layer kernel size.
      conv_stride: Conv layer stride.
      conv_padding: Conv layer padding (valid or same).
      conv_activaion: Convoultional layer activation function.
      pc_vec_dim: PC layer vector dimension.
      pc_capsules: Number of pc layer capsules.
      pc_kernel_size: PC layer kernel size.
      pc_stride: PC layer stride.
      pc_padding: PC layer padding (valid or same).
      routing_rounds: Number of rounds in routing alghorithm.
      pc_activaion: Primary caps layer activation function.
      dc_vec_dim: Digital capsules out vector dimension.
      alpha: Alpha constant, used in margin loss computation.
      m_plus: M plus constant, used in margin loss computation.
      m_minus: M minus constant, used in margin loss computation.
      decoder_layers: Sizes of decoder layers.
      decoder_loss_scale: Decoder loss scale in total loss.
      dtype_float: Type of float values.
      dtype_int: Type of int values.
    '''
    self.X = tf.placeholder(dtype_float, shape=(None,)+input_dim)

    '''
    Convolutional layer.
    '''
    self.conv = tf.layers.conv2d(self.X, filters=conv_filters,
      kernel_size=conv_kernel_size, strides=conv_stride,
      padding=conv_padding, activation=conv_activaion, name="conv_layer")

    '''
    Primary Capsule (PC) layer.
    PC layer is similar to convolutional layer.
    '''
    pc_filters = pc_vec_dim * pc_capsules
    self.pc_layer = tf.layers.conv2d(self.conv, filters=pc_filters,
      kernel_size=pc_kernel_size, strides=pc_stride, padding=pc_padding,
      activation=pc_activaion, name="primary_capsules_layer")
    num_pc_capsules = self.pc_layer.get_shape().as_list()[-2]*\
      self.pc_layer.get_shape().as_list()[-3]*pc_capsules
    self.pc_layer_out = tf.reshape(self.pc_layer, (-1, num_pc_capsules, pc_vec_dim))
    self.pc_layer_squashed = squash(self.pc_layer_out)

    '''
    Digit Capsule (DC) layer.
    '''
    self.dc_layer = FCCapsLayer(capsules=classes,
      vec_dim=dc_vec_dim, routing_rounds=routing_rounds,
      dtype_float=dtype_float)
    self.dc_layer_out = self.dc_layer(self.pc_layer_squashed,
      prev_capsules=num_pc_capsules, prev_vec_dim=pc_vec_dim)

    '''
    Results.
    '''
    self.Y = tf.placeholder(dtype_int, shape=(None))
    self.l2_norm_dc_out = tf.sqrt(tf.reduce_sum(tf.square(self.dc_layer_out),
      axis=-2) + ERROR_CONST)
    self.logit = tf.reshape(self.l2_norm_dc_out, (-1, classes))
    self.pred_label = tf.argmax(self.logit, axis=-1)
    correct = tf.nn.in_top_k(self.logit, self.Y, 1)
    self.accuracy = tf.reduce_mean(tf.cast(correct, dtype_float))

    '''
    Digit Capsule (DC) layer loss.
    '''
    T = tf.one_hot(self.Y, depth=classes)
    max_l = tf.maximum(0.0, m_plus-self.logit)
    max_r = tf.maximum(0.0, self.logit-m_minus)
    L = T*tf.square(max_l) + alpha*(1-T)*tf.square(max_r)
    self.margin_loss = tf.reduce_mean(tf.reduce_sum(L, axis=1))

    '''
    Decoder layer.
    '''
    mask = tf.reshape(T, (-1, classes))
    reconstruction_mask_reshaped = tf.reshape(
      mask, [-1, 1, classes, 1, 1])
    self.fc_input = tf.multiply(
      self.dc_layer_out, reconstruction_mask_reshaped,
      name="caps2_output_masked")
    decoder_input = tf.reshape(self.fc_input,
      [-1, classes * dc_vec_dim],
      name="decoder_input")
    # self.out = tf.squeeze(self.digit_layer_out, axis=-1)
    # self.fc_input = tf.boolean_mask(self.out, self.mask, axis=0)
    
    self.fc_layers = []
    for (i, size) in enumerate(decoder_layers):
      if i == 0:
        fc = tf.contrib.layers.fully_connected(decoder_input, num_outputs=size,
          activation_fn=tf.nn.relu)
        self.fc_layers.append(fc)
      else:
        fc = tf.contrib.layers.fully_connected(self.fc_layers[-1], num_outputs=size,
          activation_fn=tf.nn.relu)
        self.fc_layers.append(fc)

    num_outputs = input_dim[0]*input_dim[1]*input_dim[2]
    self.decoder_out = tf.contrib.layers.fully_connected(self.fc_layers[-1], num_outputs=num_outputs,
      activation_fn=tf.nn.sigmoid)

    '''
    Decoder loss.
    '''
    origin = tf.reshape(self.X, shape=(-1, num_outputs))
    self.decoder_loss = tf.reduce_mean(tf.square(self.decoder_out - origin))

    '''
    Total loss.
    '''
    self.loss = self.margin_loss + decoder_loss_scale * self.decoder_loss

class FCCapsLayer(object):
  def __init__(self,
               capsules,
               vec_dim,
               routing_rounds,
               dtype_float):
    '''
    Initializing fully connected capsule layer.
    Used for digit caps layer in current CapsNet realization.
    Args:
      capsules: Number of capsules in current layer.
      vec_dim: Len of output vector from one capsule.
      routing_rounds: Number of routing rounds in routing algorithm.
      dtype_float: Type of float values.
    '''
    self.capsules = capsules
    self.vec_dim = vec_dim
    self.routing_rounds = routing_rounds
    self.dtype_float = dtype_float

  def __call__(self, input, prev_capsules, prev_vec_dim):
    '''
    Constructing capsule layer.
    Args:
      input: Vector of vectors, input from previous capsule layer.
      prev_capsules: Number of capsules in prev layer.
      prev_vec_dim: Len of output vector from capsule from prev layer.
      batch_size: Batch size.
    '''

    '''
    U_hat computing.
    '''
    W_init = tf.random_normal((1, prev_capsules, self.capsules,
      self.vec_dim, prev_vec_dim), dtype=self.dtype_float, stddev=STDDEV)
    W = tf.Variable(W_init, name='digit_caps_weights')
    input =  tf.reshape(input,
      shape=(-1, prev_capsules, 1, prev_vec_dim, 1))
    input = tf.tile(input, [1, 1, self.capsules, 1, 1])
    W_tile = tf.tile(W, [tf.shape(input)[0], 1, 1, 1, 1])
    '''
    Matmul of last two dimensions:
    dimensions: (..., self.vec_dim, prev_vec_dim) * (..., prev_vec_dim, 1)
    result dim: (batch_size, prev_capsules, self.capsules, self.vec_dim, 1)
    '''
    U_hat = tf.matmul(W_tile, input)
    b = tf.zeros((tf.shape(input)[0], tf.shape(input)[1], self.capsules, 1, 1),
      self.dtype_float)
    return self.routing(U_hat, b)

  def routing(self, input, b):
    '''
    Implementation of routing algorithm.
    Args:
      input: U_hat vector.
      b: Initial temporary coefficient.
    '''
    input_stopped = tf.stop_gradient(input, name='stop_gradient')
    for i in range(self.routing_rounds):
      c = tf.nn.softmax(b, axis=2)
      if i == self.routing_rounds-1:
        s = tf.multiply(c, input)
        s = tf.reduce_sum(s, axis=1, keepdims=True)
        v = squash(s, axis=-2)
        return v
      elif i < self.routing_rounds-1:
        s = tf.multiply(c, input_stopped)
        s = tf.reduce_sum(s, axis=1, keepdims=True)
        v = squash(s, axis=-2)
        v_tiled = tf.tile(v, [1, tf.shape(input_stopped)[1], 1, 1, 1])
        input_produce_v  = tf.matmul(input_stopped, v_tiled,
          transpose_a=True)
        b += input_produce_v

def squash(vector, axis=-1, epsilon=ERROR_CONST):
  '''
  Squash function.
  Args:
    vector: a tensor with shape [batch_size, num_caps, caps_out_vec_len, 1] or
      [batch_size, num_caps, 1, caps_out_vec_len, 1].
    axis: Axis of finding squared_vector_norm.
    epsilon: Small constant, use to prevent numerical errors.
  Returns:
    A squashed vector with the same shape as input vector.
  '''
  squared_vector_norm = tf.reduce_sum(tf.square(vector), axis, keepdims=True)
  vector_norm = tf.sqrt(squared_vector_norm + epsilon)
  scaling_factor = squared_vector_norm / (1.0 + squared_vector_norm)
  vector_squashed = scaling_factor * vector / vector_norm
  return vector_squashed