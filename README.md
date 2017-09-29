Get the challenge data and run the DataLoader, transfer the files into a google cloud bucket.
My bucket name is ``coral-weaver-4010``, yours will be different.
````
BUCKET_NAME=coral-weaver-4010
````
To train locally but using google cloud (just to test, you must have the files locally):
```
OUTPUT_PATH=logs
DATA_PATH=DataFiles/training_tfrecords

gcloud ml-engine local train \
    --module-name trainer.task \
    --package-path trainer/ \
    --distributed \
    -- \
    --file-dir $DATA_PATH \
    --train-steps 10 \
    --num-epochs 1 \
    --learning-rate .01 \
    --job-dir $OUTPUT_PATH \
    --verbosity INFO
````

To train on google cloud using GPUs (costs money, takes time):
````
JOB_NAME=isles_train_job
BUCKET_NAME=coral-weaver-4010
OUTPUT_PATH=gs://$BUCKET_NAME/$JOB_NAME
DATA_PATH=gs://$BUCKET_NAME/data/isles_tfrecords
REGION=us-east1

gcloud ml-engine jobs submit training $JOB_NAME \
    --job-dir $OUTPUT_PATH \
    --runtime-version 1.2 \
    --module-name trainer.task \
    --package-path trainer/ \
    --scale-tier CUSTOM \
    --config config.yaml \
    --region $REGION \
    -- \
    --file-dir $DATA_PATH \
    --learning-rate .01 \
    --train-steps 1000 \
    --num-epochs 30 \
    --verbosity DEBUG
````

To view the output in tensorboard:
````
python -m tensorflow.tensorboard --logdir=$OUTPUT_PATH
````
To create a model:
````
gcloud ml-engine models create isles_2017
````
To export and deploy a model on google cloud choose a checkpoint and write:
````
export MODEL_BINARIES=$OUTPUT_PATH/export_ckpt_1504
gcloud ml-engine versions create v7 --model isles_2017 --origin $MODEL_BINARIES --runtime-version 1.2
````

To get predictions for the training set:
````
DATA_FORMAT=TF_RECORD
BUCKET_NAME=coral-weaver-4010
INPUT_PATHS=gs://$BUCKET_NAME/data/isles_tfrecords/*
MODEL_NAME=isles_2017
REGION=us-east1
now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME=isles_predict_$now
OUTPUT_PATH=gs://$BUCKET_NAME/$JOB_NAME/predictions
MAX_WORKER_COUNT=15

gcloud ml-engine jobs submit prediction $JOB_NAME \
    --model $MODEL_NAME \
    --input-paths $INPUT_PATHS \
    --output-path $OUTPUT_PATH \
    --region $REGION \
    --data-format $DATA_FORMAT \
    --max-worker-count=$MAX_WORKER_COUNT
````
Copy the output from google cloud (you may have to create a local directory first):
````
gsutil cp $OUTPUT_PATH/* DataFiles/raw_training_predictions/
````

To get the predictions for the test set run the commands above but change the input path to:
````
INPUT_PATHS=gs://$BUCKET_NAME/data/testing_tfrecords/*
````

Copy the output to your machine:
````
gsutil cp $OUTPUT_PATH/* DataFiles/raw_testing_predictions/
````
Delete the error log files and run the export file to get .nii files
