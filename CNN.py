import tensorflow as tf


class ConvolutionLayer3D(object):
    def __init__(self, layer_num, filter, strides, padding='VALID'):
        assert len(filter) == 5  # [filter_depth, filter_height, filter_width, in_channels, out_channels]
        assert len(strides) == 5  # must match input dimensions [batch, in_depth, in_height, in_width, in_channels]
        assert padding in ['VALID', 'SAME']
        self.filter = filter
        self.strides = strides
        self.padding = padding

        self.w = tf.Variable(initial_value=tf.truncated_normal(shape=filter), name='weights')
        self.b = tf.Variable(tf.constant(1.0, shape=[filter[-1]]), name='biases')

    def operate(self, input):
        convolution = tf.nn.conv3d(input, self.w, self.strides, self.padding, name=None)
        return tf.nn.relu(convolution + self.b)


class DeconvolutionLayer3D(object):
    def __init__(self, layer_num, filter, output_shape, strides, padding='VALID'):
        assert len(filter) == 5  # [depth, height, width, output_channels, in_channels]
        assert len(strides) == 5  # must match input dimensions [batch, depth, height, width, in_channels]
        assert padding in ['VALID', 'SAME']
        self.filter = filter
        self.output_shape = output_shape
        self.strides = strides
        self.padding = padding

        self.w = tf.Variable(initial_value=tf.truncated_normal(shape=filter), name='weights')
        self.b = tf.Variable(tf.constant(1.0, shape=[filter[-2]]), name='biases')

    def operate(self, input):
        deconvolution = tf.nn.conv3d_transpose(input, self.w, self.output_shape, self.strides, self.padding, name=None)
        return deconvolution + self.b
