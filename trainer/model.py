import tensorflow as tf
from VNet import v_net
from tensorflow.python.lib.io import file_io as file_io
from random import shuffle as shuffle_fn

TRAIN, EVAL, PREDICT = 'TRAIN', 'EVAL', 'PREDICT'


def dice_coefficient(volume_1, volume_2):
    with tf.variable_scope('calc_dice_coefficient'):
        intersection = tf.reduce_sum(volume_1 * volume_2)
        size_i1 = tf.norm(volume_1, ord=1)
        size_i2 = tf.norm(volume_2, ord=1)
        return 2 * intersection / (size_i1 + size_i2)


def accuracy(ground_truth, predictions):
    with tf.variable_scope('calc_accuracy'):
        return 100 * tf.reduce_mean(tf.cast(tf.equal(predictions, ground_truth), tf.float32))


def cross_entropy_loss(logits, ground_truth):
    # flatten
    logits = tf.reshape(logits, (-1,))
    ground_truth = tf.reshape(ground_truth, (-1,))
    # calculate reduce mean sigmoid cross entropy (even though all images are different size there's no problem)
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=ground_truth), name='loss')


def soft_dice_loss(logits, ground_truth):
    probabilities = tf.sigmoid(logits)
    interception_volume = tf.reduce_sum(probabilities * ground_truth)
    return - 2 * interception_volume / (tf.norm(ground_truth, ord=1) + tf.norm(probabilities, ord=1))


def mixed_loss(logits, ground_truth):
    return cross_entropy_loss(logits, ground_truth) + soft_dice_loss(logits, ground_truth)


def model_fn(mode, name, tf_input_data, tf_ground_truth, n_channels, init_learning_rate):
    logits = v_net(tf_input_data, n_channels)
    # remove expanded dims (that were only necessary for FCN)
    logits = tf.squeeze(logits)

    # predictions
    probability = tf.sigmoid(logits, name='probability')
    prediction = tf.round(probability, name='prediction')

    # exit here if predicting expand dims to emulate batch (needed for gcloud batch prediction)
    if mode == PREDICT:
        return {'name': name, 'prediction': tf.expand_dims(prediction, 0),
                'probability': tf.expand_dims(probability, 0)}

    # loss function
    with tf.variable_scope('loss_function'):
        loss = mixed_loss(logits, tf_ground_truth)

    # global step
    global_step = tf.train.get_or_create_global_step()

    dice = dice_coefficient(prediction, tf_ground_truth)
    acc = accuracy(tf_ground_truth, prediction)

    if mode == TRAIN:
        # Optimizer.
        with tf.variable_scope('optimizer'):
            learning_rate = tf.train.exponential_decay(init_learning_rate, global_step, 200, 0.95)
            train_op = tf.train.AdagradOptimizer(learning_rate).minimize(loss, global_step=global_step)

        # hook than logs training info to the console
        def formatter(d):
            return 'Step {0}: for {1} the dice coefficient is {2:.4f} and the loss is {3:.4f}' \
                .format(d[global_step], d[name], d[dice], d[loss])

        log_hook = tf.train.LoggingTensorHook([dice, loss, global_step, name], every_n_iter=1, formatter=formatter)

        tf.summary.scalar('training_dice', dice)
        tf.summary.scalar('training_loss', loss)
        train_summaries = tf.summary.merge_all()

        return train_op, log_hook, train_summaries

    if mode == EVAL:
        return {'name': name, 'dice_coefficient': dice, 'loss': loss, 'accuracy': acc, 'prediction': prediction,
                'ground_truth': tf_ground_truth}


def serving_input_fn():
    example = tf.placeholder(shape=(None, ), dtype=tf.string)
    image, _, example_name = parse_example(tf.squeeze(example))
    return image, example_name, {'example': example}


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
        # reshape
        ground_truth = tf.cast(tf.reshape(ground_truth, shape[:-1]), tf.float32)
    return image, ground_truth, example_name


def input_fn(file_dir, num_epochs=None, shuffle=False, shared_name=None):
    # get file names from given directory
    file_names = file_io.get_matching_files(file_dir[0] + '/*tfrecord')

    # shuffle if required
    if shuffle:
        shuffle_fn(file_names)

    # queue with the file names that can be shared amongst workers during training
    filename_queue = tf.FIFOQueue(100, tf.string, shared_name=shared_name)
    enque_op = filename_queue.enqueue_many([tf.train.limit_epochs(file_names, num_epochs)])
    close_op = filename_queue.close(cancel_pending_enqueues=True)

    # create queue runner and add it to queue runners
    qr = tf.train.QueueRunner(filename_queue, [enque_op], close_op,
                              queue_closed_exception_types=(tf.errors.OutOfRangeError, tf.errors.CancelledError))
    tf.train.add_queue_runner(qr)

    # read example from file
    reader = tf.TFRecordReader()
    _, example = reader.read(filename_queue)

    # parse example
    image, ground_truth, example_name = parse_example(example)

    return image, ground_truth, example_name
