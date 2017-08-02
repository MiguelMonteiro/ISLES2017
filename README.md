To run locally but using google cloud:
```
OUTPUT_PATH=logs
DATA_PATH=isles_tfrecords

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
To run using on the cloud just to test if everything is working (almost free):
````
JOB_NAME=model_export_test_6
BUCKET_NAME=coral-weaver-4010
OUTPUT_PATH=gs://$BUCKET_NAME/$JOB_NAME
DATA_PATH=gs://$BUCKET_NAME/data/isles_tfrecords
REGION=us-east1

gcloud ml-engine jobs submit training $JOB_NAME \
    --job-dir $OUTPUT_PATH \
    --runtime-version 1.2 \
    --module-name trainer.task \
    --package-path trainer/ \
    --region $REGION \
    -- \
    --file-dir $DATA_PATH \
    --learning-rate .01 \
    --train-steps 1 \
    --num-epochs 1 \
    --verbosity DEBUG
````
To run on google cloud using GPUs (expensive about 5$):
````
JOB_NAME=isles_26
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
    --train-steps 500 \
    --num-epochs 10 \
    --verbosity DEBUG
````
To get predictions from the model
````
DATA_FORMAT=TF_RECORD
BUCKET_NAME=coral-weaver-4010
INPUT_PATHS=gs://$BUCKET_NAME/data/isles_tfrecords/*
MODEL_NAME=isles_2017
VERSION_NAME=v1
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
    --data-format $DATA_FORMAT
````


