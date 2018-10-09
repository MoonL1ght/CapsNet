import numpy as np
import tensorflow as tf


'''Small constant used in squash function to avoid division by zero.'''
SMALL_CONST = 1e-7
STDDEV = 0.1
DTYPE_F = tf.float32
DTYPE_INT = tf.int32


class CapsNet(object):
  def __init__(self,
               input_dim,
               num_classes,
               routing_rounds=2,
               conv_layer=(9, 256, 1),
               conv_activaion=tf.nn.relu,
               primary_caps=(9, 32, 8, 2),
               primary_activation=None,
               digit_caps_vector_len=16,
               m_plus=0.9,
               m_minus=0.1,
               lambda_const=0.5,
               decoder_layers=(512, 1024),
               decodel_loss_scale=0.0005):
    '''
    Initializing CapsNet.
    Args:
      input_dim: size of input data, tuple.
      num_classes: total number of classes.
      routing_rounds: number of rounds in routing alghoritm. 
      conv_layer: size of convolutional layer (win_size, num_f_maps, input_maps).
      conv_activaion: convoultional layer activation function.
      primary_caps: size of primary caps layer (win_size, num_f_maps, vec_len, stride).
      primary_activation: primary caps layer activation function.
      digit_caps_vector_len: 
      m_plus:
      m_minus:
      lambda_const:
      decoder_layers:
      decodel_loss_scale:
    '''
    self.X = tf.placeholder(DTYPE_F, shape=(None,)+input_dim)
    # self.Y = tf.placeholder(DTYPE_INT, shape=(None))

    '''Creating convolutional layer.'''
    conv_kernel_size = conv_layer[0]
    conv_input_layers = input_dim[-1]
    conv_f_maps = conv_layer[1] #Number of convolutional future maps.
    conv_stride = conv_layer[2]
    # self.conv = create_conv(self.X, (conv_kernel_size, conv_kernel_size, conv_input_layers, conv_f_maps),
    #   stride=conv_stride, activation=conv_activaion, name='conv')
    # conv1_params = {
    #   "filters": 256,
    #   "kernel_size": 9,
    #   "strides": 1,
    #   "padding": "valid",
    #   "activation": tf.nn.relu,
    # }
    # self.conv1 = tf.layers.conv2d(self.X, name="conv1_", **conv1_params)
    # self.conv2 = tf.layers.conv2d(self.X, name="conv2_", **conv1_params)
    self.conv = tf.layers.conv2d(self.X, 
      filters=256, kernel_size=9, strides=1,
      padding='valid', activation=tf.nn.relu, name="conv1")
    

    '''Creating PrimaryCaps (PC) layer.
       PrimaryCaps layer is similar to convolutional layer.'''
    pc_kernel_size = primary_caps[0]
    pc_input_layers = conv_f_maps
    pc_num_capsules = primary_caps[1] #Number of PrimaryCaps layer capsules.
    pc_out_vec_len = primary_caps[2] #length of output vector from one primary capsule.
    pc_f_maps = pc_num_capsules*pc_out_vec_len #Total number of future maps of PrimaryCaps layer.
    pc_stride = primary_caps[3]
    kernel = (pc_kernel_size, pc_kernel_size, pc_input_layers, pc_f_maps)
    '''Shape of  primary_capsules: (batch_size, x_size, y_size, pc_num_capsules*pc_out_vec_len);
       x_size and y_size are like regular convolutional map sizes and depend from previous conv
       layer output size and PrimaryCaps layer stride.'''
    # self.primary_capsules = create_conv(self.conv, kernel,
    #   stride=pc_stride, activation=primary_activation, name='primary_caps')
    self.primary_capsules = tf.layers.conv2d(self.conv, 
      filters=pc_f_maps, kernel_size=9, strides=2,
      padding='valid', activation=tf.nn.relu, name="conv2")


    # self.primary_capsules_1 = self.primary_capsules
    '''Total number of regular capsules in PrimaryCaps layer:
       x_size*y_size*number_of_primary_capsules.'''
    total_num_primary_capsules = self.primary_capsules.get_shape().as_list()[-2]*\
      self.primary_capsules.get_shape().as_list()[-3]*pc_num_capsules
    '''capsules_out's shape: (batch_size, total_num_primary_capsules, pc_out_vec_len, 1).'''
    # self.primary_capsules = tf.reshape(self.primary_capsules,
    #   (-1, total_num_primary_capsules, pc_out_vec_len, 1))
    self.primary_capsules_ = tf.reshape(self.primary_capsules,
      (-1, total_num_primary_capsules, pc_out_vec_len))
    self.primary_capsules_s = squash(self.primary_capsules_)

    '''Creating DigitCaps layer.'''
    self.digit_layer = FCCapsLayer(num_classes, digit_caps_vector_len,
      routing_rounds=routing_rounds)
    self.digit_layer_out = self.digit_layer(self.primary_capsules_s,
      total_num_primary_capsules, pc_out_vec_len)

    self.Y = tf.placeholder(DTYPE_INT, shape=(None))
    self.l2_norm_digit_caps_out = tf.sqrt(tf.reduce_sum(tf.square(self.digit_layer_out),
      axis=-2) + SMALL_CONST)
    self.logit = tf.reshape(self.l2_norm_digit_caps_out, (-1, num_classes))
    self.prob = self.logit#tf.nn.softmax(self.logit, axis=-1)
    self.pred_label = tf.argmax(self.prob, axis=-1)
    self.correct = tf.nn.in_top_k(self.prob, self.Y, 1)
    self.accuracy = tf.reduce_mean(tf.cast(self.correct, tf.float32))

    self.T = tf.one_hot(self.Y, depth=num_classes)
    self.max_l = tf.maximum(0.0, m_plus-self.logit)
    self.max_r = tf.maximum(0.0, self.logit-m_minus)
    self.L = self.T*tf.square(tf.maximum(0.0, m_plus-self.logit)) +\
      lambda_const*(1-self.T)*tf.square(tf.maximum(0.0, self.logit-m_minus))
    self.ML = tf.reduce_sum(self.L, axis=1)
    self.margin_loss = tf.reduce_mean(self.ML)

    '''Decoder.'''
    self.mask = tf.reshape(self.T, (-1, num_classes))
    reconstruction_mask_reshaped = tf.reshape(
      self.mask, [-1, 1, num_classes, 1, 1],
      name="reconstruction_mask_reshaped")
    self.fc_input = tf.multiply(
      self.digit_layer_out, reconstruction_mask_reshaped,
      name="caps2_output_masked")
    self.decoder_input = tf.reshape(self.fc_input,
      [-1, num_classes * digit_caps_vector_len],
      name="decoder_input")
    # self.out = tf.squeeze(self.digit_layer_out, axis=-1)
    # self.fc_input = tf.boolean_mask(self.out, self.mask, axis=0)
    self.fc1 = tf.contrib.layers.fully_connected(self.decoder_input, num_outputs=decoder_layers[0],
      activation_fn=tf.nn.relu)
    self.fc2 = tf.contrib.layers.fully_connected(self.fc1, num_outputs=decoder_layers[1],
      activation_fn=tf.nn.relu)
    num_outputs = input_dim[0]*input_dim[1]*input_dim[2]
    self.decoded = tf.contrib.layers.fully_connected(self.fc2, num_outputs=num_outputs,
      activation_fn=tf.nn.sigmoid)

    '''Decoder loss.'''
    origin = tf.reshape(self.X, shape=(-1, num_outputs))
    self.decoder_loss = tf.reduce_mean(tf.square(self.decoded - origin))

    '''Total loss.'''
    # decodel_loss_scale = decodel_loss_scale * input_dim[0] * input_dim[1]
    self.loss = self.margin_loss + decodel_loss_scale * self.decoder_loss
 
  def explore_net(self, sess, X_batch, Y_batch):
    '''
    Returns capsnet layers sizes.
    Arguments:
      sess: Tensorflow session.
      X_bacth: Input data batch.
      Y_batch: Labels batch.
    '''
    X_size = sess.run(tf.shape(self.X), feed_dict={self.X: X_batch})
    Y_size = sess.run(tf.shape(self.Y), feed_dict={self.Y: Y_batch})
    conv_size = sess.run(tf.shape(self.conv), feed_dict={self.X: X_batch})
    primary_capsules_size = sess.run(tf.shape(self.primary_capsules), feed_dict={self.X: X_batch})
    W_size = sess.run(tf.shape(self.digit_layer.W), feed_dict={self.X: X_batch})
    U_hat_size = sess.run(tf.shape(self.digit_layer.U_hat), feed_dict={self.X: X_batch})
    digit_layer_out_size = sess.run(tf.shape(self.digit_layer_out), feed_dict={self.X: X_batch})
    fc1_size = sess.run(tf.shape(self.fc1), feed_dict={self.X: X_batch})
    fc2_size = sess.run(tf.shape(self.fc2), feed_dict={self.X: X_batch})
    decoded_size = sess.run(tf.shape(self.decoded), feed_dict={self.X: X_batch})
    return (X_size, Y_size, conv_size, primary_capsules_size, W_size, U_hat_size,
      digit_layer_out_size, fc1_size, fc2_size, decoded_size)

class FCCapsLayer(object):
  def __init__(self,
               num_capsules,
               out_vec_len,
               routing_rounds=3):
    '''
    Initializing fully connected capsule layer.
    Used for digit caps layer in current CapsNet realization.
    Args:
      num_capsules: Number of capsules in current layer.
      out_vec_len: Len of output vector from one capsule.
      prev_layer_num_capsules: Number of capsules in prev layer.
      pev_vec_len: Len of output vector from capsule from prev layer.
      routing_rounds: Number of routing rounds in routing algorithm.
    '''
    self.num_capsules=num_capsules
    self.out_vec_len=out_vec_len
    self.routing_rounds=routing_rounds
    '''Weights matrix is used in matrix multiplication by output vectors from
       capsules for previous layer.'''

  def __call__(self, input, prev_layer_num_capsules, pev_vec_len):
    '''
    Constructing capsule layer.
    Args:
      input: Vector of vectors, input from previous capsule layer.
      prev_layer_num_capsules: Number of capsules in prev layer.
      pev_vec_len: Len of output vector from capsule from prev layer.
      batch_size: Batch size.
    '''
    '''U_hat computing.'''
    W_init = tf.random_normal((1, prev_layer_num_capsules, self.num_capsules,
      self.out_vec_len, pev_vec_len),
      dtype=DTYPE_F, stddev=STDDEV)
    self.W = tf.Variable(W_init, name='digit_caps_weights')
    self.input =  tf.reshape(input,
      shape=(-1, prev_layer_num_capsules, 1, pev_vec_len, 1))
    self.input = tf.tile(self.input, [1, 1, self.num_capsules, 1, 1])
    self.W_tile = tf.tile(self.W, [tf.shape(self.input)[0], 1, 1, 1, 1])
    '''Matmul last two dimensions:
       dimensions: (..., pev_vec_len, self.out_vec_len).transpose * (..., pev_vec_len, 1),
       result dim: (batch_size, prev_layer_num_capsules, self.num_capsules, self.out_vec_len, 1)'''
    self.U_hat = tf.matmul(self.W_tile, self.input)
    self.b = tf.zeros((tf.shape(self.input)[0], tf.shape(self.input)[1], self.num_capsules, 1, 1), DTYPE_F)
    return self.routing(self.U_hat, self.b)

  def routing(self, input, b):
    '''
    Implementation of routing algorithm.
    Args:
      input: U_hat
      b: Initial temporary coefficient.
    '''
    input_stopped = tf.stop_gradient(input, name='stop_gradient')
    for i in range(self.routing_rounds):
      self.c = tf.nn.softmax(b, axis=2)
      if i == self.routing_rounds-1:
        self.s = tf.multiply(self.c, input)
        self.s = tf.reduce_sum(self.s, axis=1, keepdims=True)
        self.v = squash(self.s, axis=-2)
        return self.v
      elif i < self.routing_rounds-1:
        s = tf.multiply(self.c, input_stopped)
        s = tf.reduce_sum(s, axis=1, keepdims=True)
        v = squash(s, axis=-2)
        v_tiled = tf.tile(v, [1, tf.shape(input_stopped)[1], 1, 1, 1])
        input_produce_v = tf.reduce_sum(input_stopped * v_tiled, axis=3, keepdims=True)
        b += input_produce_v


def create_conv(input, kernel, stride=1, activation=tf.nn.relu,
                padding='VALID', name='conv_layer'):
  '''
    Create convolutional layer.
    Args:
      input

    Returns:
      A squashed vector with the same shape as input vector.
  '''
  with tf.name_scope(name):
    stride_shape = (1, stride, stride, 1)
    conv_kernel_init = tf.truncated_normal(kernel, dtype=DTYPE_F, stddev=STDDEV)
    conv_kernel = tf.Variable(conv_kernel_init, name='weights_'+name)
    conv_bias = tf.Variable(tf.constant(0.0, shape=[kernel[-1]], dtype=DTYPE_F),
      trainable=True, name='biases_'+name)
    conv = tf.nn.conv2d(input, conv_kernel, stride_shape, padding=padding)
    if activation == None:
      return tf.nn.bias_add(conv, conv_bias)
    else:
      return activation(tf.nn.bias_add(conv, conv_bias))

def squash(vector, axis=-1):
  '''
    Squash function.
    Args:
      vector: a tensor with shape [batch_size, num_caps, caps_out_vec_len, 1] or
       [batch_size, num_caps, 1, caps_out_vec_len, 1].
    Returns:
      A squashed vector with the same shape as input vector.
  '''
  squared_vector_norm = tf.reduce_sum(tf.square(vector), axis, keepdims=True)
  vector_norm = tf.sqrt(squared_vector_norm + SMALL_CONST)
  scaling_factor = squared_vector_norm / (1.0 + squared_vector_norm)
  vector_squashed = scaling_factor * vector / vector_norm
  return vector_squashed