import tensorflow as tf

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


def model_fn(n_channels):
    #n_channels = images[0].shape[-1]
    # Input data (must expand_dims because batch_size is always 1)
    tf_input_data = tf.expand_dims(tf.placeholder(tf.float32, shape=(None, None, None, n_channels), name='tf_input_data'), 0)
    tf_ground_truth = tf.expand_dims(tf.placeholder(tf.float32, shape=(None, None, None,), name='tf_ground_truth'), 0)

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
    #global_step = tf.train.get_or_create_global_step()
    global_step = tf.contrib.framework.get_or_create_global_step()
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



# max_steps = 11
# with tf.Session(graph=graph) as session:
#     tf.global_variables_initializer().run()
#     print('Initialized')
#     for step in range(max_steps):
#
#         # since all images have different sizes, we must train with batch size 1
#         pos = step % (len(ground_truths) - 1)
#         pos=0
#         image = images[pos]
#         ground_truth = ground_truths[pos]
#         # reshape to have batch dimension and channel dimensions (fo size 1 each)
#         image = add_batch_dimension(image)
#         ground_truth = add_batch_dimension(ground_truth)
#
#         # train step
#         feed_dict = {'tf_input_data:0': image, 'tf_ground_truth:0': ground_truth}
#         _, l, summary, prediction = session.run([train_step, loss, tf_summary, tf_prediction], feed_dict=feed_dict)
#         train_writer.add_summary(summary)
#
#         if step % 1 == 0:
#             print('Minibatch loss at step %d: %f' % (step, l))
#             prediction = prediction.squeeze()
#             ground_truth = ground_truth.squeeze()
#             print('The Dice Coefficient is: {0}'.format(dice_coefficient(prediction, ground_truth)))
#             print('The Hausdorff Distance is: {0}'.format(hausdorff_distance(prediction, ground_truth)))
#             print('The Average Symmetric Surface Distance is: {0}'.format(average_symmetric_surface_distance(prediction, ground_truth)))
#             print('------------------------------------------------------')
#     train_writer.flush()
