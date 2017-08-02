# JOB_NAME=crf_test_1
# BUCKET_NAME=coral-weaver-4010
# OUTPUT_PATH=gs://$BUCKET_NAME/$JOB_NAME
# DATA_PATH=gs://$BUCKET_NAME/data/isles_tfrecords
# REGION=us-east1
#
# gcloud ml-engine jobs submit training $JOB_NAME \
#     --job-dir $OUTPUT_PATH \
#     --runtime-version 1.2 \
#     --module-name trainer.task \
#     --package-path trainer/ \
#     --scale-tier CUSTOM \
#     --config config.yaml \
#     --region $REGION \
#     -- \
#     --file-dir $DATA_PATH \
#     --learning-rate .01 \
#     --train-steps 200 \
#     --num-epochs 10 \
#     --verbosity DEBUG
import os
from oauth2client.client import GoogleCredentials
from googleapiclient import discovery, errors
import logging

job_name = 'asdf'
bucket_path = 'gs://coral-weaver-4010'
data_path = os.path.join(bucket_path, 'data/isles_tfrecords')
job_dir = os.path.join(bucket_path, job_name)
training_inputs = {'scaleTier': 'CUSTOM',
                   'masterType': 'complex_model_m',
                   'workerType': 'complex_model_m',
                   'parameterServerType': 'large_model',
                   'workerCount': 9,
                   'parameterServerCount': 3,
                   'packageUris': ['gs://my/trainer/path/package-0.0.0.tar.gz'],
                   'pythonModule': 'trainer.task',
                   'args': ['--file-dir', data_path, '--learning-rate', 'value2', '--train-steps', '200',
                            '--num-epochs', '1', '--verbosity', 'debug'],
                   'region': 'us-central1',
                   'jobDir': 'gs://my/training/job/directory',
                   'runtimeVersion': '1.2'}

job_spec = {'jobId': job_name, 'trainingInput': training_inputs}

project_name = 'my_project_name'
project_id = 'projects/{}'.format(project_name)

credentials = GoogleCredentials.get_application_default()

cloudml = discovery.build('ml', 'v1', credentials=credentials)

request = cloudml.projects().jobs().create(body=job_spec, parent=project_id)
response = request.execute()

request = cloudml.projects().jobs().create(body=job_spec, parent=project_id)

try:
    response = request.execute()
    # You can put your code for handling success (if any) here.

except errors.HttpError, err:
    # Do whatever error response is appropriate for your application.
    # For this example, just send some text to the logs.
    # You need to import logging for this to work.
    logging.error('There was an error creating the training job.'
                  ' Check the details:')
    logging.error(err._get_reason())




import tensorflow as tf
from tensorflow.python.platform import gfile
with tf.Session() as sess:
    model_filename = 'export/saved_model.pb'
    with gfile.FastGFile(model_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        g_in = tf.import_graph_def(graph_def)
LOGDIR='YOUR_LOG_LOCATION'
train_writer = tf.summary.FileWriter(LOGDIR)
train_writer.add_graph(sess.graph)