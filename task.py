import argparse
import json
import os
import threading
import tensorflow as tf
import model
from tensorflow.python.saved_model import signature_constants as sig_constants
from tensorflow.python.lib.io import file_io
from six.moves import cPickle as pickle
import numpy as np


tf.logging.set_verbosity(tf.logging.INFO)


def load_data(file_path):
    images_path = 'ProcessedData/BrainArray.images.pickle'
    with file_io.FileIO(images_path, 'rb') as f:
        images = pickle.load(f)
    ground_truth_path = 'ProcessedData/BrainArray.ground_truth.pickle'
    with file_io.FileIO(ground_truth_path, 'rb') as f:
        ground_truth = pickle.load(f)
    return images, ground_truth


class EvaluationRunHook(tf.train.SessionRunHook):
    """EvaluationRunHook performs continuous evaluation of the model.
  Args:
    checkpoint_dir (string): Dir to store model checkpoints
    metric_dir (string): Dir to store metrics like accuracy and auroc
    graph (tf.Graph): Evaluation graph
    eval_frequency (int): Frequency of evaluation every n train steps
    eval_steps (int): Evaluation steps to be performed
  """

    def __init__(self, checkpoint_dir, metric_dict, graph, eval_frequency, file_path, **kwargs):

        tf.logging.set_verbosity(eval_frequency)
        self._checkpoint_dir = checkpoint_dir
        self._kwargs = kwargs
        self._eval_every = eval_frequency
        self._latest_checkpoint = None
        self._checkpoints_since_eval = 0
        self._graph = graph
        self.file_path = file_path
        # With the graph object as default graph
        # See https://www.tensorflow.org/api_docs/python/tf/Graph#as_default
        # Adds ops to the graph object
        with graph.as_default():
            value_dict, update_dict = tf.contrib.metrics.aggregate_metric_map(metric_dict)

            # Op that creates a Summary protocol buffer by merging summaries
            self._summary_op = tf.summary.merge([
                tf.summary.scalar(name, value_op)
                for name, value_op in value_dict.iteritems()
            ])

            # Saver class add ops to save and restore
            # variables to and from checkpoint
            self._saver = tf.train.Saver()

            # Creates a global step to contain a counter for
            # the global training step
            self._gs = tf.contrib.framework.get_or_create_global_step()

            self._final_ops_dict = value_dict
            self._eval_ops = update_dict.values()

        # MonitoredTrainingSession runs hooks in background threads
        # and it doesn't wait for the thread from the last session.run()
        # call to terminate to invoke the next hook, hence locks.
        self._eval_lock = threading.Lock()
        self._checkpoint_lock = threading.Lock()
        # create two file writers on for training and one for validation
        self._file_writer = tf.summary.FileWriter(os.path.join(checkpoint_dir, 'valid'))

    def after_run(self, run_context, run_values):
        # Always check for new checkpoints in case a single evaluation
        # takes longer than checkpoint frequency and _eval_every is >1
        self._update_latest_checkpoint()

        if self._eval_lock.acquire(False):
            try:
                if self._checkpoints_since_eval > self._eval_every:
                    self._checkpoints_since_eval = 0
                    self._run_eval()
            finally:
                self._eval_lock.release()

    def _update_latest_checkpoint(self):
        """Update the latest checkpoint file created in the output dir."""
        if self._checkpoint_lock.acquire(False):
            try:
                latest = tf.train.latest_checkpoint(self._checkpoint_dir)
                if not latest == self._latest_checkpoint:
                    self._checkpoints_since_eval += 1
                    self._latest_checkpoint = latest
            finally:
                self._checkpoint_lock.release()

    def end(self, session):
        """Called at then end of session to make sure we always evaluate."""
        self._update_latest_checkpoint()

        with self._eval_lock:
            self._run_eval()

    def _run_eval(self):
        """Run model evaluation and generate summaries."""
        coord = tf.train.Coordinator(clean_stop_exception_types=(
            tf.errors.CancelledError, tf.errors.OutOfRangeError))

        with tf.Session(graph=self._graph) as session:

            # Restores previously saved variables from latest checkpoint
            self._saver.restore(session, self._latest_checkpoint)

            session.run([
                tf.tables_initializer(),
                tf.local_variables_initializer()
            ])
            tf.train.start_queue_runners(coord=coord, sess=session)
            train_step = session.run(self._gs)

            tf.logging.info('Starting Evaluation For Step: {}'.format(train_step))

            with coord.stop_on_exception():
                eval_step = 0
                train_set, train_labels, valid_set, valid_labels, test_set, test_labels = load_data(self.file_path)
                valid_batch_generator = BatchGenerator(valid_set, valid_labels, 500)
                for batch_data, batch_labels in valid_batch_generator:
                    if coord.should_stop():
                        break
                    feed_dict = {'tf_input_data:0': batch_data, 'tf_labels:0': batch_labels, 'keep_prob:0': 1.0}
                    summaries, final_values, _ = session.run(
                        [self._summary_op, self._final_ops_dict, self._eval_ops], feed_dict=feed_dict)

            # Write the summaries
            self._file_writer.add_summary(summaries, global_step=train_step)
            self._file_writer.flush()
            tf.logging.info(final_values)


def run(target, is_chief, train_steps, job_dir, file_path, batch_size, eval_frequency, export_format, num_epochs):
    file_path = file_path[0]
    images, ground_truth = load_data(file_path)

    # If the server is chief which is `master`
    # In between graph replication Chief is one node in
    # the cluster with extra responsibility and by default
    # is worker task zero. We have assigned master as the chief.
    if is_chief:
        evaluation_graph = tf.Graph()
        with evaluation_graph.as_default():

            metric_dict = model.model_fn(model.EVAL)

        hooks = [EvaluationRunHook(job_dir, metric_dict, evaluation_graph, eval_frequency, file_path)]
    else:
        hooks = []

    # Create a new graph and specify that as default
    with tf.Graph().as_default():
        # Placement of ops on devices using replica device setter
        # which automatically places the parameters on the `ps` server
        # and the `ops` on the workers
        #
        # See:
        # https://www.tensorflow.org/api_docs/python/tf/train/replica_device_setter
        with tf.device(tf.train.replica_device_setter()):
            # Returns the training graph and global step tensor
            train_op, global_step_tensor = model.model_fn(model.TRAIN)

        # Creates a MonitoredSession for training
        # MonitoredSession is a Session-like object that handles
        # initialization, recovery and hooks
        # https://www.tensorflow.org/api_docs/python/tf/train/MonitoredTrainingSession
        with tf.train.MonitoredTrainingSession(master=target,
                                               is_chief=is_chief,
                                               checkpoint_dir=job_dir,
                                               hooks=hooks,
                                               save_checkpoint_secs=60,
                                               save_summaries_steps=100) as session:
            # Global step to keep track of global number of steps particularly in
            # distributed setting
            #step = global_step_tensor.eval(session=session)

            # give some random tensors because of feed_dict
            feed_dict = {'tf_input_data:0': train_set[0:2], 'tf_labels:0': train_labels[0:2], 'keep_prob:0': 1.0}
            step = session.run(global_step_tensor, feed_dict=feed_dict)

            # Run the training graph which returns the step number as tracked by
            # the global step tensor.
            # When train epochs is reached, session.should_stop() will be true. does nothing without queues
            while (train_steps is None or step < train_steps) and not session.should_stop():
                offset = (step * batch_size) % (len(train_labels) - batch_size)
                batch_data = train_set[offset:(offset + batch_size)]
                batch_labels = train_labels[offset:(offset + batch_size)]
                feed_dict = {'tf_input_data:0': batch_data, 'tf_labels:0': batch_labels, 'keep_prob:0': .9375}

                step, _ = session.run([global_step_tensor, train_op], feed_dict=feed_dict)

                if step % 250 == 0:
                    tf.logging.info('Step: {0}'.format(step))

        # Find the filename of the latest saved checkpoint file
        latest_checkpoint = tf.train.latest_checkpoint(job_dir)

        # # Only perform this if chief
        # if is_chief:
        #     build_and_run_exports(latest_checkpoint,
        #                           job_dir,
        #                           model.SERVING_INPUT_FUNCTIONS[export_format],
        #                           hidden_units)

#
# def build_and_run_exports(latest, job_dir, serving_input_fn, hidden_units):
#     """Given the latest checkpoint file export the saved model.
#   Args:
#     latest (string): Latest checkpoint file
#     job_dir (string): Location of checkpoints and model files
#     name (string): Name of the checkpoint to be exported. Used in building the
#       export path.
#     hidden_units (list): Number of hidden units
#     learning_rate (float): Learning rate for the SGD
#   """
#
#     prediction_graph = tf.Graph()
#     exporter = tf.saved_model.builder.SavedModelBuilder(
#         os.path.join(job_dir, 'export'))
#     with prediction_graph.as_default():
#         features, inputs_dict = serving_input_fn()
#         prediction_dict = model.model_fn(
#             model.PREDICT,
#             features,
#             None,  # labels
#             hidden_units=hidden_units,
#             learning_rate=None  # learning_rate unused in prediction mode
#         )
#         saver = tf.train.Saver()
#
#         inputs_info = {
#             name: tf.saved_model.utils.build_tensor_info(tensor)
#             for name, tensor in inputs_dict.iteritems()
#         }
#         output_info = {
#             name: tf.saved_model.utils.build_tensor_info(tensor)
#             for name, tensor in prediction_dict.iteritems()
#         }
#         signature_def = tf.saved_model.signature_def_utils.build_signature_def(
#             inputs=inputs_info,
#             outputs=output_info,
#             method_name=sig_constants.PREDICT_METHOD_NAME
#         )
#
#     with tf.Session(graph=prediction_graph) as session:
#         session.run([tf.local_variables_initializer(), tf.tables_initializer()])
#         saver.restore(session, latest)
#         exporter.add_meta_graph_and_variables(
#             session,
#             tags=[tf.saved_model.tag_constants.SERVING],
#             signature_def_map={
#                 sig_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature_def
#             },
#             legacy_init_op=tf.saved_model.main_op.main_op()
#         )
#
#     exporter.save()
#


def dispatch(*args, **kwargs):
    """Parse TF_CONFIG to cluster_spec and call run() method
  TF_CONFIG environment variable is available when running using
  gcloud either locally or on cloud. It has all the information required
  to create a ClusterSpec which is important for running distributed code.
  """

    tf_config = os.environ.get('TF_CONFIG')

    # If TF_CONFIG is not available run local
    if not tf_config:
        return run('', True, *args, **kwargs)

    tf_config_json = json.loads(tf_config)

    cluster = tf_config_json.get('cluster')
    job_name = tf_config_json.get('task', {}).get('type')
    task_index = tf_config_json.get('task', {}).get('index')

    # If cluster information is empty run local
    if job_name is None or task_index is None:
        return run('', True, *args, **kwargs)

    cluster_spec = tf.train.ClusterSpec(cluster)
    server = tf.train.Server(cluster_spec,
                             job_name=job_name,
                             task_index=task_index)

    # Wait for incoming connections forever
    # Worker ships the graph to the ps server
    # The ps server manages the parameters of the model.
    #
    # See a detailed video on distributed TensorFlow
    # https://www.youtube.com/watch?v=la_M6bCV91M
    if job_name == 'ps':
        server.join()
        return
    elif job_name in ['master', 'worker']:
        return run(server.target, job_name == 'master', *args, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file-path',
                        required=True,
                        type=str,
                        help='Input files local or GCS', nargs='+')
    parser.add_argument('--job-dir',
                        required=True,
                        type=str,
                        help="""\
                      GCS or local dir for checkpoints, exports, and
                      summaries. Use an existing directory to load a
                      trained model, or a new directory to retrain""")
    parser.add_argument('--train-steps',
                        type=int,
                        help='Maximum number of training steps to perform.')
    parser.add_argument('--batch-size',
                        type=int,
                        default=64,
                        help='Batch size for training steps')
    parser.add_argument('--eval-frequency',
                        default=1,
                        help='Perform one evaluation per n checkpoints (not training steps))')
    parser.add_argument('--num-epochs',
                        type=int,
                        help='Maximum number of epochs on which to train')
    parser.add_argument('--export-format',
                        type=str,
                        choices=[model.JSON, model.CSV, model.EXAMPLE],
                        default=model.JSON,
                        help="""\
                      Desired input format for the exported saved_model
                      binary.""")
    parser.add_argument('--verbosity',
                        choices=[
                            'DEBUG',
                            'ERROR',
                            'FATAL',
                            'INFO',
                            'WARN'
                        ],
                        default='INFO',
                        help='Set logging verbosity')
    parse_args, unknown = parser.parse_known_args()
    # Set python level verbosity
    tf.logging.set_verbosity(parse_args.verbosity)
    # Set C++ Graph Execution level verbosity
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(
        tf.logging.__dict__[parse_args.verbosity] / 10)
    del parse_args.verbosity

    if unknown:
        tf.logging.warn('Unknown arguments: {}'.format(unknown))

    dispatch(**parse_args.__dict__)
