import argparse
import json
import os
import tensorflow as tf
import model
from EvaluationRunHook import EvaluationRunHook
from CheckpointExporterHook import CheckpointExporterHook
from tensorflow.python.lib.io import file_io as file_io

tf.logging.set_verbosity(tf.logging.INFO)


def get_summary_dir(job_dir):
    n = 0
    while 1:
        summary_dir = job_dir+'/train_'+str(n)
        if not file_io.is_directory(summary_dir):
            return summary_dir
        n += 1


def run(cluster_spec, target, is_chief, train_steps, job_dir, train_files, eval_files, num_epochs, learning_rate):
    num_channels = 6
    hooks = list()
    # does not work well in distributed mode cause it only counts local steps (I think...)
    hooks.append(tf.train.StopAtStepHook(train_steps))

    if is_chief:
        evaluation_graph = tf.Graph()
        with evaluation_graph.as_default():
            # Features and label tensors
            image, ground_truth, name = model.input_fn(eval_files, 1, shuffle=False, shared_name=None)
            # Returns dictionary of tensors to be evaluated
            metric_dict = model.model_fn(model.EVAL, name, image, ground_truth, num_channels, learning_rate)
            # hook that performs evaluation separate from training
            hooks.append(EvaluationRunHook(job_dir, metric_dict, evaluation_graph))
        hooks.append(CheckpointExporterHook(job_dir))

    # Create a new graph and specify that as default
    with tf.Graph().as_default():
        with tf.device(tf.train.replica_device_setter(cluster=cluster_spec)):

            # Features and label tensors as read using filename queue
            image, ground_truth, name = model.input_fn(train_files, num_epochs, shuffle=True, shared_name='train_queue')

            # Returns the training graph and global step tensor
            train_op, log_hook, train_summaries = model.model_fn(model.TRAIN, name, image, ground_truth,
                                                                 num_channels, learning_rate)
            # Hook that logs training to the console
            hooks.append(log_hook)

            train_summary_hook = tf.train.SummarySaverHook(save_steps=1, output_dir=get_summary_dir(job_dir),
                                                           summary_op=train_summaries)
            hooks.append(train_summary_hook)

        # Creates a MonitoredSession for training
        # MonitoredSession is a Session-like object that handles
        # initialization, recovery and hooks
        # https://www.tensorflow.org/api_docs/python/tf/train/MonitoredTrainingSession
        with tf.train.MonitoredTrainingSession(master=target,
                                               is_chief=is_chief,
                                               checkpoint_dir=job_dir,
                                               hooks=hooks,
                                               save_checkpoint_secs=60*3,
                                               save_summaries_steps=1,
                                               log_step_count_steps=5) as session:
            # Run the training graph which returns the step number as tracked by
            # the global step tensor.
            # When train epochs is reached, session.should_stop() will be true.
            while not session.should_stop():
                session.run(train_op)


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
    parser.add_argument('--train-files',
                        required=True,
                        type=str,
                        help='training input files or directory local or GCS',
                        nargs='+')
    parser.add_argument('--eval-files',
                        required=True,
                        type=str,
                        help='Eval input files or directory local or GCS',
                        nargs='+')
    parser.add_argument('--job-dir',
                        required=True,
                        type=str,
                        help='GCS or local dir for checkpoints, exports, and '
                             'summaries. Use an existing directory to load a '
                             'trained model, or a new directory to retrain')
    parser.add_argument('--train-steps',
                        type=int,
                        help='Maximum number of training steps to perform.')
    parser.add_argument('--num-epochs',
                        type=int,
                        help='Maximum number of epochs on which to train')
    parser.add_argument('--learning-rate',
                        type=float,
                        default=1e-2,
                        help='Initial learning rate')
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
