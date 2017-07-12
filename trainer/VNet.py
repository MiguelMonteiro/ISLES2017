import tensorflow as tf
from Layers import convolution_layer_3d, deconvolution_layer_3d

# interpretation of Vnet
def v_net(tf_input, n_channels):
    # down path level 1
    #n_channels = 16
    # convolution 5x5x5 filter with stride 1 (keeps same dimension as input)
    c1 = tf.nn.relu(convolution_layer_3d(tf_input, [5, 5, 5, n_channels, n_channels], [1, 1, 1, 1, 1]))
    # sum with input (part of learning residual function)
    c11 = c1 + tf_input
    # use convolution for down sampling instead of pooling (double channels half dimension)
    c12 = tf.nn.relu(convolution_layer_3d(c11, [2, 2, 2, n_channels, n_channels * 2], [1, 2, 2, 2, 1]))

    # down path level 2 (only difference is that it uses more convolution layers)
    c2 = tf.nn.relu(convolution_layer_3d(c12, [5, 5, 5, n_channels * 2, n_channels * 2], [1, 1, 1, 1, 1]))
    c2 = tf.nn.relu(convolution_layer_3d(c2, [5, 5, 5, n_channels * 2, n_channels * 2], [1, 1, 1, 1, 1]))
    c21 = c2 + c12
    c22 = tf.nn.relu(convolution_layer_3d(c21, [2, 2, 2, n_channels * 2, n_channels * 4], [1, 2, 2, 2, 1]))

    # down path level 3
    c3 = tf.nn.relu(convolution_layer_3d(c22, [5, 5, 5, n_channels * 4, n_channels * 4], [1, 1, 1, 1, 1]))
    c3 = tf.nn.relu(convolution_layer_3d(c3, [5, 5, 5, n_channels * 4, n_channels * 4], [1, 1, 1, 1, 1]))
    c3 = tf.nn.relu(convolution_layer_3d(c3, [5, 5, 5, n_channels * 4, n_channels * 4], [1, 1, 1, 1, 1]))
    c31 = c3 + c22
    c32 = tf.nn.relu(convolution_layer_3d(c31, [2, 2, 2, n_channels * 4, n_channels * 8], [1, 2, 2, 2, 1]))

    # down path level 4
    c4 = tf.nn.relu(convolution_layer_3d(c32, [5, 5, 5, n_channels * 8, n_channels * 8], [1, 1, 1, 1, 1]))
    c4 = tf.nn.relu(convolution_layer_3d(c4, [5, 5, 5, n_channels * 8, n_channels * 8], [1, 1, 1, 1, 1]))
    c4 = tf.nn.relu(convolution_layer_3d(c4, [5, 5, 5, n_channels * 8, n_channels * 8], [1, 1, 1, 1, 1]))
    c41 = c4 + c32
    c42 = tf.nn.relu(convolution_layer_3d(c41, [2, 2, 2, n_channels * 8, n_channels * 16], [1, 2, 2, 2, 1]))

    # down path level 5 (last level)
    c5 = tf.nn.relu(convolution_layer_3d(c42, [5, 5, 5, n_channels * 16, n_channels * 16], [1, 1, 1, 1, 1]))
    c5 = tf.nn.relu(convolution_layer_3d(c5, [5, 5, 5, n_channels * 16, n_channels * 16], [1, 1, 1, 1, 1]))
    c5 = tf.nn.relu(convolution_layer_3d(c5, [5, 5, 5, n_channels * 16, n_channels * 16], [1, 1, 1, 1, 1]))
    c51 = c5 + c42
    # start up-sampling
    c52 = tf.nn.relu(deconvolution_layer_3d(c51, [2, 2, 2, n_channels * 8, n_channels * 16], tf.shape(c41), [1, 2, 2, 2, 1]))

    # up-path level 5
    d5 = tf.concat((c52, c41), axis=-1)

    d51 = tf.nn.relu(convolution_layer_3d(d5, [5, 5, 5, n_channels * 16, n_channels * 8], [1, 1, 1, 1, 1]))
    d51 = tf.nn.relu(convolution_layer_3d(d51, [5, 5, 5, n_channels * 8, n_channels * 8], [1, 1, 1, 1, 1]))
    d51 = tf.nn.relu(convolution_layer_3d(d51, [5, 5, 5, n_channels * 8, n_channels * 8], [1, 1, 1, 1, 1]))

    d52 = d51 + c52
    d53 = tf.nn.relu(deconvolution_layer_3d(d52, [2, 2, 2, n_channels * 4, n_channels * 8], tf.shape(c31), [1, 2, 2, 2, 1]))

    # up-path level 4
    d4 = tf.concat((d53, c31), axis=-1)

    d41 = tf.nn.relu(convolution_layer_3d(d4, [5, 5, 5, n_channels * 8, n_channels * 4], [1, 1, 1, 1, 1]))
    d41 = tf.nn.relu(convolution_layer_3d(d41, [5, 5, 5, n_channels * 4, n_channels * 4], [1, 1, 1, 1, 1]))
    d41 = tf.nn.relu(convolution_layer_3d(d41, [5, 5, 5, n_channels * 4, n_channels * 4], [1, 1, 1, 1, 1]))

    d42 = d41 + d53
    d43 = tf.nn.relu(deconvolution_layer_3d(d42, [2, 2, 2, n_channels * 2, n_channels * 4], tf.shape(c21), [1, 2, 2, 2, 1]))

    # up-path level 3
    d3 = tf.concat((d43, c21), axis=-1)
    d31 = tf.nn.relu(convolution_layer_3d(d3, [5, 5, 5, n_channels * 4, n_channels * 2], [1, 1, 1, 1, 1]))
    d31 = tf.nn.relu(convolution_layer_3d(d31, [5, 5, 5, n_channels * 2, n_channels * 2], [1, 1, 1, 1, 1]))

    d32 = d31 + d43
    d33 = tf.nn.relu(deconvolution_layer_3d(d32, [2, 2, 2, n_channels, n_channels * 2], tf.shape(c11), [1, 2, 2, 2, 1]))

    # up-path level 2
    d2 = tf.concat((d33, c11), axis=-1)
    d21 = tf.nn.relu(convolution_layer_3d(d2, [5, 5, 5, n_channels * 2, n_channels], [1, 1, 1, 1, 1]))
    d22 = d21 + d33

    # output 1x1x1 conv to give outputs, with 2 channels (classification result) and same dimension as input (no relu last layer)
    logits = convolution_layer_3d(d22, [1, 1, 1, n_channels, 1], [1, 1, 1, 1, 1])

    return logits
