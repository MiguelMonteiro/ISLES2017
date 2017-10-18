import os
from PostProcessing.Utils import MorphologicalFilter, adjust_training_data

os.chdir('../')
prediction_dir = 'DataFiles/raw_training_predictions'
image_dir = 'DataFiles/training_tfrecords'
# model search morphological filter
thresholds = [.5, .6, .7, .8, .9, 1.0]
output = []
for threshold in thresholds:
    mf = MorphologicalFilter(threshold)
    print('Threshold: {0}'.format(str(threshold)))
    metrics = adjust_training_data(mf, prediction_dir, image_dir)
    output.append([threshold, metrics])

