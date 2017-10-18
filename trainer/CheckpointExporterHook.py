import tensorflow as tf
import os
import threading
from tensorflow.python.saved_model import signature_constants as sig_constants
from tensorflow.python.ops import variables
from tensorflow.python.ops import lookup_ops
import model
import re


def my_main_op():
    init_local = variables.local_variables_initializer()
    init_tables = lookup_ops.tables_initializer()
    return tf.group(init_local, init_tables)


class CheckpointExporterHook(tf.train.SessionRunHook):
    def __init__(self, checkpoint_dir, export_every=1):

        self._checkpoint_dir = checkpoint_dir
        self._export_every = export_every  # eval every x checkpoints
        self._latest_checkpoint = None
        self._checkpoints_since_export = 0

        # MonitoredTrainingSession runs hooks in background threads
        # and it doesn't wait for the thread from the last session.run()
        # call to terminate to invoke the next hook, hence locks.
        self._export_lock = threading.Lock()
        self._checkpoint_lock = threading.Lock()

    def after_run(self, run_context, run_values):
        # Always check for new checkpoints in case a single evaluation
        # takes longer than checkpoint frequency and _eval_every is >1
        self._update_latest_checkpoint()

        if self._export_lock.acquire(False):
            try:
                if self._checkpoints_since_export >= self._export_every:
                    self._checkpoints_since_export = 0
                    self._run_export()
            finally:
                self._export_lock.release()

    def _update_latest_checkpoint(self):
        """Update the latest checkpoint file created in the output dir."""
        if self._checkpoint_lock.acquire(False):
            try:
                latest = tf.train.latest_checkpoint(self._checkpoint_dir)
                if not latest == self._latest_checkpoint:
                    self._checkpoints_since_export += 1
                    self._latest_checkpoint = latest
            finally:
                self._checkpoint_lock.release()

    def end(self, session):
        self._update_latest_checkpoint()

        with self._export_lock:
            self._run_export()

    def _run_export(self):

        export_dir = 'export_ckpt_' + re.findall('\d+', self._latest_checkpoint)[-1]
        tf.logging.info('Exporting model from checkpoint {0}'.format(self._latest_checkpoint))
        prediction_graph = tf.Graph()
        try:
            exporter = tf.saved_model.builder.SavedModelBuilder(os.path.join(self._checkpoint_dir, export_dir))
        except IOError:
            tf.logging.info('Checkpoint {0} already exported, continuing...'.format(self._latest_checkpoint))
            return

        with prediction_graph.as_default():
            image, name, inputs_dict = model.serving_input_fn()
            prediction_dict = model.model_fn(model.PREDICT, name, image, None, 6, None)

            saver = tf.train.Saver()

            inputs_info = {name: tf.saved_model.utils.build_tensor_info(tensor)
                           for name, tensor in inputs_dict.iteritems()}

            output_info = {name: tf.saved_model.utils.build_tensor_info(tensor)
                           for name, tensor in prediction_dict.iteritems()}

            signature_def = tf.saved_model.signature_def_utils.build_signature_def(
                inputs=inputs_info,
                outputs=output_info,
                method_name=sig_constants.PREDICT_METHOD_NAME
            )

        with tf.Session(graph=prediction_graph) as session:
            saver.restore(session, self._latest_checkpoint)
            exporter.add_meta_graph_and_variables(
                session,
                tags=[tf.saved_model.tag_constants.SERVING],
                signature_def_map={sig_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature_def},
                legacy_init_op=my_main_op()
            )

        exporter.save()
