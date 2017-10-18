from sklearn.model_selection import KFold
import subprocess
import os
import numpy as np
import time

data_generator = KFold(n_splits=5, shuffle=True, random_state=13)

data_format = 'TF_RECORD'
model_name = 'isles_2017_cv'
max_worker_count = str(15)
base_job_name = 'isles_cv_prediction'
bucket_name = 'gs://coral-weaver-4010'
data_dir = os.path.join(bucket_name, 'data/isles_tfrecords')
region = 'us-east1'

file_paths = subprocess.check_output('gsutil ls ' + data_dir, shell=True)
file_paths = np.array(file_paths.split('\n')[1:-1])

np.random.seed(1)
np.random.shuffle(file_paths)

fold = 1
for train_indices, valid_indices in data_generator.split(file_paths):
    eval_file_paths = ','.join(file_paths[valid_indices])
    output_path = os.path.join(bucket_name, base_job_name, 'fold_' + str(fold))
    job_name = base_job_name + '_fold_' + str(fold)
    version = 'fold_' + str(fold)
    fold += 1

    command = 'gcloud ml-engine jobs submit prediction {0} ' \
              '--model {1} ' \
              '--version {2} ' \
              '--input-paths {3} ' \
              '--output-path {4} ' \
              '--region {5} ' \
              '--data-format {6} ' \
              '--max-worker-count {7}'.format(job_name, model_name, version, eval_file_paths, output_path, region,
                                              data_format, max_worker_count)

    print command
    output = subprocess.call(command, shell=True)
    time.sleep(2700)
