import tensorflow as tf


def xavier_normal_dist(shape):
    return tf.truncated_normal(shape, mean=0, stddev=tf.sqrt(3. / shape[-1] + shape[-2]))


def convolution_layer_3d(layer_input, filter, strides, padding='SAME'):
    assert len(filter) == 5  # [filter_depth, filter_height, filter_width, in_channels, out_channels]
    assert len(strides) == 5  # must match input dimensions [batch, in_depth, in_height, in_width, in_channels]
    assert padding in ['VALID', 'SAME']
    w = tf.Variable(initial_value=tf.truncated_normal(shape=filter), name='weights')

    #w = tf.Variable(initial_value=xavier_normal_dist(shape=filter), name='weights')

    b = tf.Variable(tf.constant(1.0, shape=[filter[-1]]), name='biases')
    convolution = tf.nn.conv3d(layer_input, w, strides, padding)
    return convolution + b


def deconvolution_layer_3d(layer_input, filter, output_shape, strides, padding='SAME'):
    assert len(filter) == 5  # [depth, height, width, output_channels, in_channels]
    assert len(strides) == 5  # must match input dimensions [batch, depth, height, width, in_channels]
    assert padding in ['VALID', 'SAME']
    w = tf.Variable(initial_value=tf.truncated_normal(shape=filter), name='weights')
    #w = tf.Variable(initial_value=xavier_normal_dist(shape=filter), name='weights')
    b = tf.Variable(tf.constant(1.0, shape=[filter[-2]]), name='biases')
    deconvolution = tf.nn.conv3d_transpose(layer_input, w, output_shape, strides, padding)
    return deconvolution + b


def max_pooling_3d(layer_input, ksize, strides, padding='VALID'):
    assert len(ksize) == 5  # [batch, depth, rows, cols, channels]
    assert len(strides) == 5  # [batch, depth, rows, cols, channels]
    assert ksize[0] == ksize[4]
    assert ksize[0] == 1
    assert strides[0] == strides[4]
    assert strides[0] == 1
    return tf.nn.max_pool3d(layer_input, ksize, strides, padding)
