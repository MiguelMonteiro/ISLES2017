import tensorflow as tf
from Layers import convolution_layer_3d, deconvolution_layer_3d, prelu


# n_channels = 16
# interpretation of Vnet
def v_net(tf_input, n_channels):

    with tf.variable_scope('encoder_level_1'):
        # convolution 5x5x5 filter with stride 1 (keeps same dimension as input)
        with tf.variable_scope('convolution_1'):
            c1 = prelu(convolution_layer_3d(tf_input, [5, 5, 5, n_channels, n_channels], [1, 1, 1, 1, 1]))
        # sum with input (part of learning residual function)
        c11 = c1 + tf_input
        # use convolution for down sampling instead of pooling (double channels half dimension)
        with tf.variable_scope('down_sample'):
            c12 = prelu(convolution_layer_3d(c11, [2, 2, 2, n_channels, n_channels * 2], [1, 2, 2, 2, 1]))

    with tf.variable_scope('encoder_level_2'):
        with tf.variable_scope('convolution_1'):
            c2 = prelu(convolution_layer_3d(c12, [5, 5, 5, n_channels * 2, n_channels * 2], [1, 1, 1, 1, 1]))
        with tf.variable_scope('convolution_2'):
            c2 = prelu(convolution_layer_3d(c2, [5, 5, 5, n_channels * 2, n_channels * 2], [1, 1, 1, 1, 1]))
        c21 = c2 + c12
        with tf.variable_scope('down_sample'):
            c22 = prelu(convolution_layer_3d(c21, [2, 2, 2, n_channels * 2, n_channels * 4], [1, 2, 2, 2, 1]))

    with tf.variable_scope('encoder_level_3'):
        with tf.variable_scope('convolution_1'):
            c3 = prelu(convolution_layer_3d(c22, [5, 5, 5, n_channels * 4, n_channels * 4], [1, 1, 1, 1, 1]))
        with tf.variable_scope('convolution_2'):
            c3 = prelu(convolution_layer_3d(c3, [5, 5, 5, n_channels * 4, n_channels * 4], [1, 1, 1, 1, 1]))
        with tf.variable_scope('convolution_3'):
            c3 = prelu(convolution_layer_3d(c3, [5, 5, 5, n_channels * 4, n_channels * 4], [1, 1, 1, 1, 1]))
        c31 = c3 + c22
        with tf.variable_scope('down_sample'):
            c32 = prelu(convolution_layer_3d(c31, [2, 2, 2, n_channels * 4, n_channels * 8], [1, 2, 2, 2, 1]))

    with tf.variable_scope('encoder_level_4'):
        with tf.variable_scope('convolution_1'):
            c4 = prelu(convolution_layer_3d(c32, [5, 5, 5, n_channels * 8, n_channels * 8], [1, 1, 1, 1, 1]))
        with tf.variable_scope('convolution_2'):
            c4 = prelu(convolution_layer_3d(c4, [5, 5, 5, n_channels * 8, n_channels * 8], [1, 1, 1, 1, 1]))
        with tf.variable_scope('convolution_3'):
            c4 = prelu(convolution_layer_3d(c4, [5, 5, 5, n_channels * 8, n_channels * 8], [1, 1, 1, 1, 1]))
        c41 = c4 + c32
        with tf.variable_scope('down_sample'):
            c42 = prelu(convolution_layer_3d(c41, [2, 2, 2, n_channels * 8, n_channels * 16], [1, 2, 2, 2, 1]))

    with tf.variable_scope('encoder_decoder_level_5'):
        with tf.variable_scope('convolution_1'):
            c5 = prelu(convolution_layer_3d(c42, [5, 5, 5, n_channels * 16, n_channels * 16], [1, 1, 1, 1, 1]))
        with tf.variable_scope('convolution_2'):
            c5 = prelu(convolution_layer_3d(c5, [5, 5, 5, n_channels * 16, n_channels * 16], [1, 1, 1, 1, 1]))
        with tf.variable_scope('convolution_3'):
            c5 = prelu(convolution_layer_3d(c5, [5, 5, 5, n_channels * 16, n_channels * 16], [1, 1, 1, 1, 1]))
        c51 = c5 + c42
        # start up-sampling
        with tf.variable_scope('up_sample'):
            c52 = prelu(deconvolution_layer_3d(c51, [2, 2, 2, n_channels * 8, n_channels * 16], tf.shape(c41), [1, 2, 2, 2, 1]))

    with tf.variable_scope('decoder_level_4'):
        # up-path level 5
        d4 = tf.concat((c52, c41), axis=-1)

        with tf.variable_scope('convolution_1'):
            d41 = prelu(convolution_layer_3d(d4, [5, 5, 5, n_channels * 16, n_channels * 8], [1, 1, 1, 1, 1]))
        with tf.variable_scope('convolution_2'):
            d41 = prelu(convolution_layer_3d(d41, [5, 5, 5, n_channels * 8, n_channels * 8], [1, 1, 1, 1, 1]))
        with tf.variable_scope('convolution_3'):
            d41 = prelu(convolution_layer_3d(d41, [5, 5, 5, n_channels * 8, n_channels * 8], [1, 1, 1, 1, 1]))

        d42 = d41 + c52
        with tf.variable_scope('up_sample'):
            d43 = prelu(deconvolution_layer_3d(d42, [2, 2, 2, n_channels * 4, n_channels * 8], tf.shape(c31), [1, 2, 2, 2, 1]))

    with tf.variable_scope('decoder_level_3'):
        # up-path level 4
        d3 = tf.concat((d43, c31), axis=-1)

        with tf.variable_scope('convolution_1'):
            d31 = prelu(convolution_layer_3d(d3, [5, 5, 5, n_channels * 8, n_channels * 4], [1, 1, 1, 1, 1]))
        with tf.variable_scope('convolution_2'):
            d31 = prelu(convolution_layer_3d(d31, [5, 5, 5, n_channels * 4, n_channels * 4], [1, 1, 1, 1, 1]))
        with tf.variable_scope('convolution_3'):
            d31 = prelu(convolution_layer_3d(d31, [5, 5, 5, n_channels * 4, n_channels * 4], [1, 1, 1, 1, 1]))

        d32 = d31 + d43
        with tf.variable_scope('up_sample'):
            d33 = prelu(deconvolution_layer_3d(d32, [2, 2, 2, n_channels * 2, n_channels * 4], tf.shape(c21), [1, 2, 2, 2, 1]))

    with tf.variable_scope('decoder_level_2'):
        d2 = tf.concat((d33, c21), axis=-1)
        with tf.variable_scope('convolution_1'):
            d21 = prelu(convolution_layer_3d(d2, [5, 5, 5, n_channels * 4, n_channels * 2], [1, 1, 1, 1, 1]))
        with tf.variable_scope('convolution_2'):
            d21 = prelu(convolution_layer_3d(d21, [5, 5, 5, n_channels * 2, n_channels * 2], [1, 1, 1, 1, 1]))

        d22 = d21 + d33
        with tf.variable_scope('up_sample'):
            d23 = prelu(deconvolution_layer_3d(d22, [2, 2, 2, n_channels, n_channels * 2], tf.shape(c11), [1, 2, 2, 2, 1]))

    with tf.variable_scope('decoder_level_1'):
        # up-path level 2
        d1 = tf.concat((d23, c11), axis=-1)
        with tf.variable_scope('convolution_1'):
            d11 = prelu(convolution_layer_3d(d1, [5, 5, 5, n_channels * 2, n_channels], [1, 1, 1, 1, 1]))
        d12 = d11 + d23

    # output 1x1x1 conv to give outputs, with 2 channels (classification result) and same dimension as input (no relu last layer)
    with tf.variable_scope('output_layer'):
        logits = convolution_layer_3d(d12, [1, 1, 1, n_channels, 1], [1, 1, 1, 1, 1])

    return logits
