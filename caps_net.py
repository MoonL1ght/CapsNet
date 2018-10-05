import numpy as np
import tensorflow as tf


'''Small constant used in squash function to avoid division by zero.'''
SQAUSH_CONST = 1e-10
STDDEV = 0.1
DTYPE_F = tf.float32
DTYPE_INT = tf.int32


class CapsNet(object):
  def __init__(self,
               input_dim,
               num_classes,
               routing_rounds=3,
               conv_layer=(3, 64, 1),
               conv_activaion=tf.nn.relu,
               primary_caps=(3, 32, 8, 2),
               primary_activation=None,
               digit_caps_vector_len=16,
               m_plus=0.9,
               m_minus=0.1,
               lambda_const=0.5,
               decoder_layers=(256, 512),
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
    self.Y = tf.placeholder(DTYPE_INT, shape=(None))

    '''Creating convolutional layer.'''
    conv_kernel_size = conv_layer[0]
    conv_input_layers = input_dim[-1]
    conv_f_maps = conv_layer[1] #Number of convolutional future maps.
    conv_stride = conv_layer[2]
    self.conv = create_conv(self.X, (conv_kernel_size, conv_kernel_size, conv_input_layers, conv_f_maps),
      stride=conv_stride, activation=conv_activaion, name='conv')

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
    self.primary_capsules = create_conv(self.conv, kernel,
      stride=pc_stride, activation=primary_activation, name='primary_caps')
    '''Total number of regular capsules in PrimaryCaps layer:
       x_size*y_size*number_of_primary_capsules.'''
    total_num_primary_capsules = self.primary_capsules.get_shape().as_list()[-2]*\
      self.primary_capsules.get_shape().as_list()[-3]*pc_num_capsules
    '''capsules_out's shape: (batch_size, total_num_primary_capsules, pc_out_vec_len, 1).'''
    self.primary_capsules = tf.reshape(self.primary_capsules,
      (-1, total_num_primary_capsules, pc_out_vec_len, 1))
    self.primary_capsules = squash(self.primary_capsules)

    '''Creating DigitCaps layer.'''
    self.digit_layer = FCCapsLayer(num_classes, digit_caps_vector_len,
      routing_rounds=routing_rounds)
    self.digit_layer_out = self.digit_layer(self.primary_capsules,
      total_num_primary_capsules, pc_out_vec_len)

    l2_norm_digit_caps_out = tf.sqrt(tf.reduce_sum(tf.square(self.digit_layer_out), axis=-2))
    self.logit = tf.reshape(l2_norm_digit_caps_out, (-1, num_classes))
    self.prob = tf.nn.softmax(self.logit, axis=-1)
    self.pred_label = tf.argmax(self.prob, axis=-1)
    self.correct = tf.nn.in_top_k(self.prob, self.Y, 1)
    self.accuracy = tf.reduce_mean(tf.cast(self.correct, tf.float32))

    T = tf.one_hot(self.Y, depth=num_classes)
    L = T*tf.square(tf.maximum(0.0, m_plus-self.logit)) +\
      lambda_const*(1-T)*tf.square(tf.maximum(0.0, self.logit-m_minus))
    self.margin_loss = tf.reduce_mean(tf.reduce_sum(L, axis=1))

    '''Decoder.'''
    mask = tf.reshape(T, (-1, num_classes))
    out = tf.squeeze(self.digit_layer_out, axis=-1)
    fc_input = tf.boolean_mask(out, mask, axis=0)
    self.fc1 = tf.contrib.layers.fully_connected(fc_input, num_outputs=decoder_layers[0],
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
    decodel_loss_scale = decodel_loss_scale * input_dim[0] * input_dim[1]
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
    W_init = tf.truncated_normal((1, prev_layer_num_capsules, self.num_capsules,
      pev_vec_len, self.out_vec_len),
      dtype=DTYPE_F, stddev=STDDEV)
    W = tf.Variable(W_init, name='digit_caps_weights')
    input =  tf.reshape(input,
      shape=(-1, prev_layer_num_capsules, 1, pev_vec_len, 1))
    input = tf.tile(input, [1, 1, self.num_capsules, 1, 1])
    W_tile = tf.tile(W, [tf.shape(input)[0], 1, 1, 1, 1])
    '''Matmul last two dimensions:
       dimensions: (..., pev_vec_len, self.out_vec_len).transpose * (..., pev_vec_len, 1),
       result dim: (batch_size, prev_layer_num_capsules, self.num_capsules, self.out_vec_len, 1)'''
    self.U_hat = tf.matmul(W_tile, input, transpose_a=True)
    b = tf.zeros((tf.shape(input)[0], tf.shape(input)[1], self.num_capsules, 1, 1), DTYPE_F)
    return tf.squeeze(self.routing(self.U_hat, b), axis=1)

  def routing(self, input, b):
    '''
    Implementation of routing algorithm.
    Args:
      input: U_hat
      b: Initial temporary coefficient.
    '''
    input_stopped = tf.stop_gradient(input, name='stop_gradient')
    for i in range(self.routing_rounds):
      c = tf.nn.softmax(b, axis=2)
      if i == self.routing_rounds-1:
        s = tf.multiply(c, input)
        s = tf.reduce_sum(s, axis=1, keepdims=True)
        v = squash(s)
        return v
      elif i < self.routing_rounds-1:
        s = tf.multiply(c, input_stopped)
        s = tf.reduce_sum(s, axis=1, keepdims=True)
        v = squash(s)
        v_tiled = tf.tile(v, [1, tf.shape(input_stopped)[1], 1, 1, 1])
        input_produce_v = tf.reduce_sum(input_stopped * v_tiled, axis=3, keepdims=True)
        b += input_produce_v


def create_conv(input, kernel, stride=1, activation=tf.nn.relu,
                padding='SAME', name='conv_layer'):
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

def squash(vector):
  '''
    Squash function.
    Args:
      vector: a tensor with shape [batch_size, num_caps, caps_out_vec_len, 1] or
       [batch_size, num_caps, 1, caps_out_vec_len, 1].
    Returns:
      A squashed vector with the same shape as input vector.
  '''
  squared_vector_norm = tf.reduce_sum(tf.square(vector), -2, keepdims=True)
  vector_norm = tf.sqrt(squared_vector_norm + SQAUSH_CONST)
  scaling_factor = squared_vector_norm / (1 + squared_vector_norm)
  vector_squashed = scaling_factor * vector / (vector_norm)
  return vector_squashed