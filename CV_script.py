from sklearn.model_selection import KFold
import subprocess
import os
import numpy as np

data_generator = KFold(n_splits=5, shuffle=True, random_state=13)
base_job_name = 'cv_test_3'
bucket_name = 'gs://coral-weaver-4010'
data_dir = os.path.join(bucket_name, 'data/isles_tfrecords')
region = 'us-east1'

file_paths = subprocess.check_output('gsutil ls ' + data_dir, shell=True)
file_paths = np.array(file_paths.split('\n')[1:-1])

np.random.seed(1)
np.random.shuffle(file_paths)


fold = 1
for train_indices, valid_indices in data_generator.split(file_paths):
    train_file_paths = ' '.join(file_paths[train_indices])
    eval_file_paths = ' '.join(file_paths[valid_indices])
    job_dir = os.path.join(bucket_name, base_job_name, 'fold_' + str(fold))
    job_name = base_job_name + '_fold_' + str(fold)
    fold += 1

    # command = 'gcloud ml-engine jobs submit training {0} ' \
    #           '--job-dir {1} ' \
    #           '--runtime-version 1.2 ' \
    #           '--module-name trainer.task ' \
    #           '--package-path trainer/ ' \
    #           '--scale-tier CUSTOM ' \
    #           '--config conif.yaml ' \
    #           '--region {2} ' \
    #           '-- ' \
    #           '--train-files {3} ' \
    #           '--eval-files {4} ' \
    #           '--learning-rate .01 ' \
    #           '--train-steps 1 ' \
    #           '--num-epochs 1 ' \
    #           '--verbosity DEBUG'.format(job_name, job_dir, region, train_file_paths, eval_file_paths)

    command = 'gcloud ml-engine jobs submit training {0} ' \
              '--job-dir {1} ' \
              '--runtime-version 1.2 ' \
              '--module-name trainer.task ' \
              '--package-path trainer/ ' \
              '--region {2} ' \
              '-- ' \
              '--train-files {3} ' \
              '--eval-files {4} ' \
              '--learning-rate .01 ' \
              '--train-steps 1 ' \
              '--num-epochs 1 ' \
              '--verbosity DEBUG'.format(job_name, job_dir, region, train_file_paths, eval_file_paths)

    print command
    output = subprocess.call(command, shell=True)