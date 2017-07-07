import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from tensorflow.python.lib.io import file_io
from CNN import ConvolutionLayer3D, DeconvolutionLayer3D
from metrics import dice_coefficient, hausdorff_distance, average_symmetric_surface_distance


def add_batch_dimension(image):
    new_shape = [1] + list(image.shape)
    return image.reshape(new_shape)

output_path = 'logs'


def model_fn():
    n_channels = images[0].shape[-1]
    # Input data
    tf_input_data = tf.placeholder(tf.float32, shape=(1, None, None, None, n_channels), name='tf_input_data')
    tf_ground_truth = tf.placeholder(tf.float32, shape=(1, None, None, None,), name='tf_ground_truth')

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

    def caculate_loss(logits, ground_truth):
        with tf.variable_scope('loss_function'):
            # reshape for sigmoid cross entropy
            shape = tf.shape(logits)
            flat = shape[1]*shape[2]*shape[3]*shape[4]
            l = tf.reshape(logits, (flat, 1))
            t = tf.reshape(tf_ground_truth, (flat, 1))
            # calculate reduce mean sigmoid cross entropy (even though all images are different size there's no problem)
            return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=l, labels=t), name='loss')

    loss = caculate_loss(logits, tf_ground_truth)
    # Optimizer.
    with tf.variable_scope('optimizer'):
        global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(0.05, global_step, 10000, 0.95)
        train_step = tf.train.AdagradOptimizer(learning_rate).minimize(loss, global_step=global_step,
                                                                       name='train_step')

    # Predictions for the training, validation, and test data.
    tf_prediction = tf.round(tf.nn.sigmoid(logits, name='prediction'))

    tf.summary.scalar('metrics/loss', loss)
    tf.summary.scalar('metrics/global_step', global_step)
    tf.summary.scalar('metrics/learning_rate', learning_rate)
    #tf.summary.scalar('metrics/accuracy', accuracy)

    #tf_summary = tf.summary.merge_all()





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
