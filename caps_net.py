import numpy as np
import tensorflow as tf

#Small constant used in squash function to avoid division by zero.
SQAUSH_CONST = 1e-10
CONV_STDDEV = 0.1
DTYPE_F = tf.float32
DTYPE_INT = tf.int32

class CapsNet(object):
  def __init__(self,
               input_dim,
               conv_layer=(3, 16, 1),
               conv_activaion=tf.nn.relu,
               primary_caps=(3, 16, 8, 2)):
    '''
    Initialize CapsNet.
    Args:
      input_dim: size of input data, tuple.
    '''
    self.X = tf.placeholder(DTYPE_F, shape=(None,)+input_dim)
    self.Y = tf.placeholder(DTYPE_INT, shape=(None))

    #Creating convolutional layer.
    conv_kernel_size = conv_layer[0]
    conv_input_layers = input_dim[-1]
    conv_f_maps = conv_layer[1] #Number of convolutional future maps.
    conv_stride = conv_layer[2]
    conv = create_conv(x, (conv_kernel_size, conv_kernel_size, conv_input_layers, conv_f_maps),
      stride=conv_stride, activation=conv_activaion, name='conv')

    #Creating PrimaryCaps (PC) layer.
    #PrimaryCaps layer is similar to convolutional layer.
    pc_kernel_size = primary_caps[0]
    pc_input_layers = conv[-1]
    pc_num_capsules = primary_caps[1] #Number of PrimaryCaps layer capsules.
    pc_out_vec_dim = primary_caps[2] #Dimension of output vector from one primary capsule.
    pc_f_maps = pc_num_capsules*pc_out_vec_dim #Total number of future maps of PrimaryCaps layer.
    pc_stride = primary_caps[3]
    primary_capsules = create_conv(conv, (pc_kernel_size, pc_kernel_size, pc_input_layers, pc_f_maps),
      [1, pc_stride, pc_stride, 1], 'primary_caps')
    # sss = tf.shape(capsules)[-2]*tf.shape(capsules)[-3]*NUM_CAPSULES
    # capsules_out = tf.reshape(capsules, (-1, sss, VEC_LEN, 1))






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
    stride = (1, stride, stride, 1)
    conv_kernel_init = tf.truncated_normal(kernel, dtype=DTYPE_F, stddev=CONV_STDDEV)
    conv_kernel = tf.Variable(conv_kernel_init, name='weights_'+name)
    conv_bias = tf.Variable(tf.constant(0.0, shape=[kernel_shape[-1]], dtype=DTYPE_F),
      trainable=True, name='biases_'+name)
    conv = tf.nn.conv2d(inpt, conv_kernel, stride, padding=padding)
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
  vector_norm = tf.sqrt(squared_vector_norm)
  scaling_factor = squared_vector_norm / (1 + squared_vector_norm)
  vector_squashed = scaling_factor * vector / (vector_norm + SQAUSH_CONST)
  return vector_squashed








mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

print(x_train.shape)
print(y_train)


x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
y = tf.placeholder(tf.int32, shape=(None))

def create_conv(x, kernel_shape, stride, name):
  with tf.name_scope(name):
    conv_kernel_init = tf.truncated_normal(kernel_shape, dtype=tf.float32, stddev=0.1)
    conv_kernel = tf.Variable(conv_kernel_init, name='weights_'+name)
    conv_bias = tf.Variable(tf.constant(0.0, shape=[kernel_shape[-1]], dtype=tf.float32),
      trainable=True, name='biases_'+name)
    conv = tf.nn.conv2d(x, conv_kernel, stride, padding='SAME')
    out = tf.nn.relu(tf.nn.bias_add(conv, conv_bias))
    return out

def squash(vector):
    '''Squash function
    Args:
        vector: A tensor with shape [batch_size, num_caps, vec_len, 1].
    Returns:
        A squashed vector with the same shape as vector.
    c - small constant used to avoid division by zero.
    '''
    c = 1e-10
    squared_vector_norm = tf.reduce_sum(tf.square(vector), -2, keepdims=True)
    vector_norm = tf.sqrt(squared_vector_norm)
    scaling_factor = squared_vector_norm / (1 + squared_vector_norm)
    vector_squashed = scaling_factor * vector / (vector_norm + c)
    return vector_squashed
conv1 = create_conv(x, [3, 3, 1, 32], [1, 1, 1, 1], 'conv1')

BATCH_SIZE = 3

KERNEL_SIZE = 3
NUM_CAPSULES = 16
VEC_LEN = 8
VEC_LEN_J = 12
STRIDE = 2
LABELS = 10
capsules = create_conv(conv1, [KERNEL_SIZE, KERNEL_SIZE, 32, NUM_CAPSULES*VEC_LEN],
  [1, STRIDE, STRIDE, 1], 'PrimaryCaps')


sss = tf.shape(capsules)[-2]*tf.shape(capsules)[-3]*NUM_CAPSULES
capsules_out = tf.reshape(capsules, (-1, sss, VEC_LEN, 1))

sq_capsules_out = squash(capsules_out)

capsules_out_1 = tf.reshape(sq_capsules_out, shape=(-1, sss, 1, tf.shape(sq_capsules_out)[-2], 1))
# print(tf.shape(sq_capsules_out))
input = capsules_out_1


b_IJ = tf.zeros([BATCH_SIZE, tf.shape(input)[1], LABELS, 1, 1], np.float32)

W_init = tf.truncated_normal((1, 3136, LABELS, VEC_LEN, VEC_LEN_J), dtype=tf.float32, stddev=0.1)
W = tf.Variable(W_init, name='weights_Digit')

input_1 = tf.tile(input, [1, 1, LABELS, 1, 1])
w_tile = tf.tile(W, [BATCH_SIZE, 1, 1, 1, 1])
# u_hat = tf.reduce_sum(W * input_1, axis=3, keepdims=True)
# u_hat = tf.reshape(u_hat, shape=[-1, 3136, LABELS, VEC_LEN_J, 1])
u_h = tf.matmul(w_tile, input_1, transpose_a=True)
# W = tf.get_variable('Weight', shape=[1, input_shape[1], num_dims * num_outputs] + input_shape[-2:],
#                         dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=cfg.stddev))
ROUT_ITERS = 3
for r in range(ROUT_ITERS):
  c_IJ = tf.nn.softmax(b_IJ, axis=2)
  s_J = tf.multiply(c_IJ, u_h)
  s_j = tf.reduce_sum(s_J, axis=1, keepdims=True)
  v_J = squash(s_j)
  v_J_tiled = tf.tile(v_J, [1, 3136, 1, 1, 1])
  u_produce_v = tf.reduce_sum(u_h * v_J_tiled, axis=3, keepdims=True)
  b_IJ += u_produce_v

logit = tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(v_J), axis=-2)), (-1, LABELS)) # v l2 norm
prob = tf.nn.softmax(logit, axis=-2)
pred_label = tf.argmax(prob, axis=-2)


#caps_loss
m_plus = 0.9
m_minus = 0.1
alpha = 0.5

T_c = tf.one_hot(y, depth=LABELS)
L_c = T_c*tf.maximum(0.0, m_plus-logit)**2 + alpha*(1-T_c)*tf.maximum(0.0, logit-m_minus)**2
margin_loss = tf.reduce_mean(tf.reduce_mean(L_c, axis=1))

#decoder
oh = tf.reshape(T_c, (-1, LABELS))
vv = tf.reshape(v_J, (-1, LABELS, VEC_LEN_J))
inpt_1 = tf.reshape(T_c, (-1, LABELS,1))*vv
inpt = tf.boolean_mask(vv, oh, axis=0)
fc1 = tf.contrib.layers.fully_connected(inpt, num_outputs=512)
fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs=1024)
decoded = tf.contrib.layers.fully_connected(fc2, num_outputs=28 * 28 * 1,
  activation_fn=tf.sigmoid)

#decoder_loss
# decoded = tf.reshape(decoded, (-1, 28, 28, 1))
orgin = tf.reshape(x, shape=(-1, 28*28*1))
reconstruction_err = tf.reduce_mean(tf.square(decoded - orgin))

loss = margin_loss + 0.0005 * reconstruction_err


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

x_batch = np.reshape(x_train[0:3], (3, 28, 28, 1))
y_batch = y_train[0:3]
print(x_batch.shape)

conv1_res = sess.run(conv1, feed_dict={x: x_batch})
print(conv1_res[0].shape)

caps = sess.run(capsules, feed_dict={x: x_batch})
capsules_res = sess.run(capsules_out_1, feed_dict={x: x_batch})
sq_capsules_res = sess.run(sq_capsules_out, feed_dict={x: x_batch})
print(caps[0].shape)
print(capsules_res[0].shape)
print(sq_capsules_res[0].shape)

print('====')

print(sess.run(tf.shape(conv1), feed_dict={x: x_batch}))
# print(sess.run(tf.shape(W), feed_dict={x: x_batch}))
# print(sess.run(tf.shape(input), feed_dict={x: x_batch}))
print(sess.run(tf.shape(v_J), feed_dict={x: x_batch}))

print(sess.run(tf.shape(logit), feed_dict={x: x_batch}))
print(sess.run(logit, feed_dict={x: x_batch}))
# print(sess.run(prob, feed_dict={x: x_batch}))

# print(sess.run(pred_label, feed_dict={x: x_batch}))

print('-----')
print(sess.run(L_c, feed_dict={x: x_batch, y: y_batch}))
print(sess.run(margin_loss, feed_dict={x: x_batch, y: y_batch}))

print('-=-=-=-=')
print(sess.run(vv, feed_dict={x: x_batch}))
print(sess.run(T_c, feed_dict={x: x_batch, y: y_batch}))
print(sess.run(inpt, feed_dict={x: x_batch, y: y_batch}))
print(sess.run(inpt_1, feed_dict={x: x_batch, y: y_batch}))
print(sess.run(decoded, feed_dict={x: x_batch, y: y_batch}))
print(sess.run(reconstruction_err, feed_dict={x: x_batch, y: y_batch}))
print(sess.run(loss, feed_dict={x: x_batch, y: y_batch}))
