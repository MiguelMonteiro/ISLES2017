import tensorflow as tf
import multiprocessing

TRAIN, EVAL, PREDICT = 'TRAIN', 'EVAL', 'PREDICT'
CSV, EXAMPLE, JSON = 'CSV', 'EXAMPLE', 'JSON'
PREDICTION_MODES = [CSV, EXAMPLE, JSON]


def convolution_layer_3d(layer_num, layer_input, filter, strides, padding='VALID'):
    assert len(filter) == 5  # [filter_depth, filter_height, filter_width, in_channels, out_channels]
    assert len(strides) == 5  # must match input dimensions [batch, in_depth, in_height, in_width, in_channels]
    assert padding in ['VALID', 'SAME']
    layer_name = 'Conv3D_' + str(layer_num)
    with tf.variable_scope(layer_name):
        w = tf.Variable(initial_value=tf.truncated_normal(shape=filter), name='weights')
        b = tf.Variable(tf.constant(1.0, shape=[filter[-1]]), name='biases')
        convolution = tf.nn.conv3d(layer_input, w, strides, padding)
        return convolution + b


def deconvolution_layer_3d(layer_num, layer_input, filter, output_shape, strides, padding='VALID'):
    assert len(filter) == 5  # [depth, height, width, output_channels, in_channels]
    assert len(strides) == 5  # must match input dimensions [batch, depth, height, width, in_channels]
    assert padding in ['VALID', 'SAME']
    layer_name = 'Deconv3D_' + str(layer_num)
    with tf.variable_scope(layer_name):
        w = tf.Variable(initial_value=tf.truncated_normal(shape=filter), name='weights')
        b = tf.Variable(tf.constant(1.0, shape=[filter[-2]]), name='biases')
        deconvolution = tf.nn.conv3d_transpose(layer_input, w, output_shape, strides, padding)
        return deconvolution + b


def max_pooling_3d(layer_num, layer_input, ksize, strides, padding='VALID'):
    assert len(ksize) == 5  # [batch, depth, rows, cols, channels]
    assert len(strides) == 5  # [batch, depth, rows, cols, channels]
    assert ksize[0] == ksize[4]
    assert ksize[0] == 1
    assert strides[0] == strides[4]
    assert strides[0] == 1
    layer_name = 'MaxPooling3D' + str(layer_num)
    with tf.variable_scope(layer_name):
        return tf.nn.max_pool3d(layer_input, ksize, strides, padding)


def dice_coefficient(input1, input2):
    with tf.variable_scope('dice_coefficient'):
        intersection = tf.reduce_sum(input1 * input2)
        size_i1 = tf.reduce_sum(input1)
        size_i2 = tf.reduce_sum(input2)
        return 2 * intersection / (size_i1 + size_i2)


def accuracy(ground_truth, predictions):
    with tf.variable_scope('accuracy'):
        return 100 * tf.reduce_mean(tf.cast(tf.equal(predictions, ground_truth), tf.float32))


def cross_entropy_loss(logits, ground_truth):
    # flatten
    logits = tf.reshape(logits, (-1, ))
    ground_truth = tf.reshape(ground_truth, (-1,))
    # calculate reduce mean sigmoid cross entropy (even though all images are different size there's no problem)
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=ground_truth), name='loss')


def dice_loss(logits, ground_truth):
    return -dice_coefficient(logits, ground_truth)


def model_fn(tf_input_data, tf_ground_truth, n_channels):
    # architecture
    c1 = tf.nn.relu(convolution_layer_3d(1, tf_input_data, [5, 5, 5, n_channels, 32], [1, 1, 1, 1, 1]))
    # c1 = max_pooling_3d(1, c1, [1, 3, 3, 3, 1], [1, 2, 2, 2, 1])

    c2 = tf.nn.relu(convolution_layer_3d(2, c1, [5, 5, 5, 32, 64], [1, 1, 1, 1, 1]))
    # c2 = max_pooling_3d(2, c2, [1, 3, 3, 3, 1], [1, 2, 2, 2, 1])

    d1 = tf.nn.relu(deconvolution_layer_3d(1, c2, [5, 5, 5, 32, 64], tf.shape(c1), [1, 1, 1, 1, 1]))

    logits = deconvolution_layer_3d(1, d1, [5, 5, 5, 1, 32], tf.shape(tf_ground_truth), [1, 1, 1, 1, 1])

    # remove expanded dims (that were only necessary for FCN)
    logits = tf.squeeze(logits)
    tf_ground_truth = tf.squeeze(tf_ground_truth)

    # loss function
    with tf.variable_scope('loss_function'):
        loss = cross_entropy_loss(logits, tf_ground_truth)
        #loss = dice_loss(logits, tf_ground_truth)

    # global step
    global_step = tf.train.get_or_create_global_step()

    # Optimizer.
    with tf.variable_scope('optimizer'):
        learning_rate = tf.train.exponential_decay(0.05, global_step, 10000, 0.95)
        train_op = tf.train.AdagradOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # Predictions for the training, validation, and test data.
    with tf.variable_scope('prediction'):
        tf_prediction = tf.round(tf.sigmoid(logits))

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('learning_rate', learning_rate)

    dice = dice_coefficient(tf_prediction, tf_ground_truth)
    tf.summary.scalar('dice_coefficient', dice)

    tf.summary.scalar('accuracy', accuracy(tf_ground_truth, tf_prediction))

    return train_op, global_step, dice


def parse_example(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'shape': tf.FixedLenFeature([], tf.string),
            'img_raw': tf.FixedLenFeature([], tf.string),
            'gt_raw': tf.FixedLenFeature([], tf.string)
        })

    with tf.variable_scope('decoder'):
        shape = tf.decode_raw(features['shape'], tf.int32)
        image = tf.decode_raw(features['img_raw'], tf.float64)
        ground_truth = tf.decode_raw(features['gt_raw'], tf.uint8)

    with tf.variable_scope('image'):
        # reshape and add 0 dimension (would be batch dimension)
        image = tf.cast(tf.expand_dims(tf.reshape(image, shape), 0), tf.float32)
    with tf.variable_scope('ground_truth'):
        # reshape and add 0 dimension (would be batch dimension) and add last dimension (would be channel dimension)
        ground_truth = tf.cast(tf.expand_dims(tf.expand_dims(tf.reshape(ground_truth, shape[:-1]), 0), -1), tf.float32)
    return image, ground_truth


def input_fn(filenames, num_epochs=None, shuffle=True):
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=shuffle)
    reader = tf.TFRecordReader()
    _, example = reader.read(filename_queue)
    image, ground_truth = parse_example(example)
    return image, ground_truth
