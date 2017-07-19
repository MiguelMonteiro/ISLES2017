import tensorflow as tf
import numpy as np
import os
import threading
from volume_metrics import hd, assd, obj_assd, dc


class Logger(object):
    """Logging in tensorboard without tensorflow ops. The original author of this class is Michael Gygli, 
    I deleted the logging of images and added the logging of random variables"""

    def __init__(self, log_dir):
        self.writer = tf.summary.FileWriter(log_dir)

    def log_scalar(self, tag, value, step):
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def log_random_variable(self, tag, var, step):

        self.log_scalar(tag+'/mean', np.mean(var), step)
        self.log_scalar(tag + '/std', np.std(var), step)
        self.log_scalar(tag + '/max', np.max(var), step)
        self.log_scalar(tag + '/min', np.min(var), step)
        self.writer.flush()

    def log_histogram(self, tag, values, step, bins=1000):
        # Create histogram using numpy
        values = np.array(values)
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill fields of histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(np.shape(values)))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
        # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
        # Thus, we drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()


def list_of_dict_to_dict_of_lists(list_of_dict):
    dict_of_list = {key: [] for key in list_of_dict[0].keys()}
    for el in list_of_dict:
        for key in dict_of_list.keys():
            dict_of_list[key].append(el[key])
    return dict_of_list


class EvaluationRunHook(tf.train.SessionRunHook):
    def __init__(self, checkpoint_dir, metric_dict, graph, eval_every=1):

        self._checkpoint_dir = checkpoint_dir
        self._eval_every = eval_every  # eval every x checkpoints
        self._latest_checkpoint = None
        self._checkpoints_since_eval = 0
        self._graph = graph

        # With the graph object as default graph
        # See https://www.tensorflow.org/api_docs/python/tf/Graph#as_default
        # Adds ops to the graph object
        with graph.as_default():
            # Saver class add ops to save and restore
            # variables to and from checkpoint
            self._saver = tf.train.Saver()

            # Creates a global step to contain a counter for
            # the global training step
            self._gs = tf.train.get_or_create_global_step()

            self._metric_dict = metric_dict

        # MonitoredTrainingSession runs hooks in background threads
        # and it doesn't wait for the thread from the last session.run()
        # call to terminate to invoke the next hook, hence locks.
        self._eval_lock = threading.Lock()
        self._checkpoint_lock = threading.Lock()
        self.logger = Logger(os.path.join(checkpoint_dir, 'eval'))

    def after_run(self, run_context, run_values):
        # Always check for new checkpoints in case a single evaluation
        # takes longer than checkpoint frequency and _eval_every is >1
        self._update_latest_checkpoint()

        if self._eval_lock.acquire(False):
            try:
                if self._checkpoints_since_eval >= self._eval_every:
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

            list_of_metric_dicts = []
            print('Starting evaluation')
            with coord.stop_on_exception():
                while not coord.should_stop():
                    list_of_metric_dicts.append(session.run(self._metric_dict))

            dict_of_metrics = list_of_dict_to_dict_of_lists(list_of_metric_dicts)

            # pop prediction and ground truth to calculate Hausdorff and ASSD and dice again (just as a check)
            predictions = dict_of_metrics.pop('prediction')
            ground_truths = dict_of_metrics.pop('ground_truth')
            _hd, _assd, _obj_assd, _dice = [], [], [], []
            for prediction, ground_truth in predictions, ground_truths:
                _hd.append(hd(prediction, ground_truth))
                _assd.append(assd(prediction, ground_truth))
                _obj_assd.append(obj_assd(prediction, ground_truth))
                _dice.append(dc(prediction, ground_truth))
            # put them back in
            dict_of_metrics['Hausdorff_distance'] = _hd
            dict_of_metrics['average_symetric_surface_distance'] = _assd
            # dict_of_metrics['obj_average_symetric_surface_distance'] = _obj_assd
            dict_of_metrics['external_dice'] = _dice

            # Save histogram, mean and std for each variable
            for key, value in dict_of_metrics.iteritems():
                self.logger.log_histogram(tag=key, values=value, step=train_step, bins=15)
                self.logger.log_random_variable(tag='eval_'+key, var=value, step=train_step)
            print('Finished evaluation')
