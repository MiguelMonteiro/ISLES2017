import tensorflow as tf
import os
import threading
import numpy as np
from Logger import Logger


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
            self._gs = tf.contrib.framework.get_or_create_global_step()

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
            # Save histogram, mean and std for each variable
            for key, value in dict_of_metrics.iteritems():
                self.logger.log_histogram(tag=key, values=value, step=train_step, bins=15)
                self.logger.log_random_variable(tag='eval_'+key, var=value, step=train_step)
            print('Finished evaluation')
