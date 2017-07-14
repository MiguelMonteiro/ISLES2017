import tensorflow as tf
from Layers import convolution_layer_3d, deconvolution_layer_3d
from VNet import v_net

TRAIN, EVAL, PREDICT = 'TRAIN', 'EVAL', 'PREDICT'
CSV, EXAMPLE, JSON = 'CSV', 'EXAMPLE', 'JSON'
PREDICTION_MODES = [CSV, EXAMPLE, JSON]


def jaccard_similarity_coefficient(volume_1, volume_2):
    with tf.variable_scope('jaccard_similarity_coefficient'):
        return tf.reduce_sum(tf.minimum(volume_1, volume_2)) / tf.reduce_sum(tf.maximum(volume_1, volume_2))


def dice_coefficient(volume_1, volume_2):
    with tf.variable_scope('dice_coefficient'):
        intersection = tf.reduce_sum(volume_1 * volume_2)
        size_i1 = tf.norm(volume_1, ord=1)
        size_i2 = tf.norm(volume_2, ord=1)
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


def soft_dice_loss(logits, ground_truth):
    probabilities = tf.sigmoid(logits)
    interception_volume = tf.reduce_sum(probabilities * ground_truth)
    return - 2 * interception_volume / (tf.norm(ground_truth, ord=1) + tf.norm(probabilities, ord=1))


def model_fn(tf_input_data, tf_ground_truth, n_channels):

    logits = v_net(tf_input_data, n_channels)

    # remove expanded dims (that were only necessary for FCN)
    logits = tf.squeeze(logits)
    tf_ground_truth = tf.squeeze(tf_ground_truth)

    # loss function
    with tf.variable_scope('loss_function'):
        #loss = cross_entropy_loss(logits, tf_ground_truth)
        loss = soft_dice_loss(logits, tf_ground_truth)

    # global step
    global_step = tf.train.get_or_create_global_step()

    # # Optimizer.
    with tf.variable_scope('optimizer'):
        learning_rate = tf.train.exponential_decay(5e-3, global_step, 64, 0.95)
        train_op = tf.train.AdagradOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # use gradient clipping o avoid exploding gradients
    # with tf.variable_scope('optimizer'):
    #     learning_rate = tf.train.exponential_decay(.05, global_step, 100, 0.95)
    #     optimizer = tf.train.AdagradOptimizer(learning_rate)
    #     gradients, variables = zip(*optimizer.compute_gradients(loss))
    #     gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
    #     train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)

    # Predictions for the training, validation, and test data.
    with tf.variable_scope('prediction'):
        tf_prediction = tf.round(tf.sigmoid(logits))

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('learning_rate', learning_rate)

    dice = dice_coefficient(tf_prediction, tf_ground_truth)
    tf.summary.scalar('dice_coefficient', dice)

    tf.summary.scalar('accuracy', accuracy(tf_ground_truth, tf_prediction))

    return train_op, global_step, dice, loss


def parse_example(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'shape': tf.FixedLenFeature([], tf.string),
            'img_raw': tf.FixedLenFeature([], tf.string),
            'gt_raw': tf.FixedLenFeature([], tf.string),
            'example_name': tf.FixedLenFeature([], tf.string)
        })

    with tf.variable_scope('decoder'):
        shape = tf.decode_raw(features['shape'], tf.int32)
        image = tf.decode_raw(features['img_raw'], tf.float32)
        ground_truth = tf.decode_raw(features['gt_raw'], tf.uint8)
        example_name = features['example_name']

    with tf.variable_scope('image'):
        # reshape and add 0 dimension (would be batch dimension)
        image = tf.expand_dims(tf.reshape(image, shape), 0)
    with tf.variable_scope('ground_truth'):
        # reshape and add 0 dimension (would be batch dimension) and add last dimension (would be channel dimension)
        ground_truth = tf.cast(tf.expand_dims(tf.expand_dims(tf.reshape(ground_truth, shape[:-1]), 0), -1), tf.float32)
    return image, ground_truth, example_name


def input_fn(filenames, num_epochs=None, shuffle=True):
    # using shared name ensures each worker takes a different example from the queue each time
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=shuffle)
    reader = tf.TFRecordReader()
    _, example = reader.read(filename_queue)

    image, ground_truth, example_name = parse_example(example)

    queue = tf.FIFOQueue(50, [tf.float32, tf.float32, tf.string])
    enqueue_op = queue.enqueue([image, ground_truth, example_name])

    qr = tf.train.QueueRunner(queue, [enqueue_op] * 2)
    tf.train.add_queue_runner(qr)

    return queue.dequeue()
