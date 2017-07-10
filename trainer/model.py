import tensorflow as tf
import multiprocessing

TRAIN, EVAL, PREDICT = 'TRAIN', 'EVAL', 'PREDICT'
CSV, EXAMPLE, JSON = 'CSV', 'EXAMPLE', 'JSON'
PREDICTION_MODES = [CSV, EXAMPLE, JSON]


class ConvolutionLayer3D(object):
    def __init__(self, layer_num, filter, strides, padding='VALID'):
        assert len(filter) == 5  # [filter_depth, filter_height, filter_width, in_channels, out_channels]
        assert len(strides) == 5  # must match input dimensions [batch, in_depth, in_height, in_width, in_channels]
        assert padding in ['VALID', 'SAME']
        self.layer_name = 'Conv3D_' + str(layer_num)
        self.filter = filter
        self.strides = strides
        self.padding = padding
        with tf.variable_scope(self.layer_name):
            self.w = tf.Variable(initial_value=tf.truncated_normal(shape=filter), name='weights')
            self.b = tf.Variable(tf.constant(1.0, shape=[filter[-1]]), name='biases')

    def operate(self, input):
        with tf.variable_scope(self.layer_name + '/'):
            convolution = tf.nn.conv3d(input, self.w, self.strides, self.padding, name=None)
            return tf.nn.relu(convolution + self.b)


class DeconvolutionLayer3D(object):
    def __init__(self, layer_num, filter, output_shape, strides, padding='VALID'):
        assert len(filter) == 5  # [depth, height, width, output_channels, in_channels]
        assert len(strides) == 5  # must match input dimensions [batch, depth, height, width, in_channels]
        assert padding in ['VALID', 'SAME']
        self.layer_name = 'Deconv3D_' + str(layer_num)
        self.filter = filter
        self.output_shape = output_shape
        self.strides = strides
        self.padding = padding
        with tf.variable_scope(self.layer_name):
            self.w = tf.Variable(initial_value=tf.truncated_normal(shape=filter), name='weights')
            self.b = tf.Variable(tf.constant(1.0, shape=[filter[-2]]), name='biases')

    def operate(self, input):
        with tf.variable_scope(self.layer_name + '/'):
            deconvolution = tf.nn.conv3d_transpose(input, self.w, self.output_shape, self.strides, self.padding, name=None)
            return deconvolution + self.b


def dice_coefficient(input1, input2):
    with tf.variable_scope('dice_coefficient'):
        input1 = tf.squeeze(input1)
        input2 = tf.squeeze(input2)
        intersection = tf.cast(tf.count_nonzero(input1 * input2), tf.float32)
        size_i1 = tf.count_nonzero(input1)
        size_i2 = tf.count_nonzero(input2)
        return 2 * intersection / tf.cast(size_i1 + size_i2, tf.float32)


def accuracy(ground_truth, predictions):
    with tf.variable_scope('accuracy'):
        return 100 * tf.reduce_mean(tf.cast(tf.equal(tf.squeeze(predictions), tf.squeeze(ground_truth)), tf.float32))


def caculate_loss(logits, ground_truth):
    with tf.variable_scope('loss_function'):
        # reshape for sigmoid cross entropy
        shape = tf.shape(logits)
        flat = shape[1]*shape[2]*shape[3]*shape[4]
        l = tf.reshape(logits, (flat, 1))
        t = tf.reshape(ground_truth, (flat, 1))
        # calculate reduce mean sigmoid cross entropy (even though all images are different size there's no problem)
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=l, labels=t), name='loss')


def model_fn(tf_input_data, tf_ground_truth, n_channels):
    #n_channels = images[0].shape[-1]
    # Input data (must expand_dims because batch_size is always 1)
    #tf_input_data = tf.expand_dims(tf.placeholder(tf.float32, shape=(None, None, None, n_channels), name='tf_input_data'), 0)
    #tf_ground_truth = tf.expand_dims(tf.placeholder(tf.float32, shape=(None, None, None,), name='tf_ground_truth'), 0)

    tf_input_data = tf.expand_dims(tf_input_data, 0)
    tf_ground_truth = tf.expand_dims(tf_ground_truth, 0)

    # architecture
    l1 = ConvolutionLayer3D(1, [5, 5, 5, n_channels, 64], [1, 1, 1, 1, 1])
    output_shape = tf.shape(tf.expand_dims(tf_ground_truth, -1))
    l4 = DeconvolutionLayer3D(3, [5, 5, 5, 1, 64], output_shape, [1, 1, 1, 1, 1])
    layers = [l1, l4]

    # Model
    def model(data):
        for layer in layers:
            data = layer.operate(data)

        return data

    # Training computation.
    logits = model(tf_input_data)

    loss = caculate_loss(logits, tf_ground_truth)
    global_step = tf.train.get_or_create_global_step()
    # Optimizer.
    with tf.variable_scope('optimizer'):
        learning_rate = tf.train.exponential_decay(0.05, global_step, 10000, 0.95)
        train_op = tf.train.AdagradOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # Predictions for the training, validation, and test data.
    with tf.variable_scope('prediction'):
        tf_prediction = tf.round(tf.nn.sigmoid(logits, name='prediction'))

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('learning_rate', learning_rate)
    tf.summary.scalar('dice_coefficient', dice_coefficient(tf_prediction, tf_ground_truth))
    tf.summary.scalar('accuracy', accuracy(tf_ground_truth, tf_prediction))

    return train_op, global_step


def parse_example(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'shape': tf.FixedLenFeature([], tf.string),
            'img_raw': tf.FixedLenFeature([], tf.string),
            'gt_raw': tf.FixedLenFeature([], tf.string)
        })
    shape = tf.decode_raw(features['shape'], tf.int32)
    image = tf.decode_raw(features['img_raw'], tf.float64)
    ground_truth = tf.decode_raw(features['gt_raw'], tf.uint8)

    # reshape
    image = tf.reshape(image, shape)
    ground_truth = tf.reshape(ground_truth, shape[:-1])
    return tf.cast(image, tf.float32), tf.cast(ground_truth, tf.float32)


def input_fn(filenames, num_epochs=None, shuffle=True):

    filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=shuffle)
    reader = tf.TFRecordReader()
    _, example = reader.read(filename_queue)

    # Parse the CSV File
    image, ground_truth = parse_example(example)

    return image, ground_truth