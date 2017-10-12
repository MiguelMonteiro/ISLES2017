from PostProcessing.Utils import CRF, MorphologicalFilter, export_data
import os

os.chdir('DataFiles/')

if not os.path.exists('training_export_raw'):
    os.mkdir('training_export_raw')
export_data('raw_training_predictions', 'Training', 'training_tfrecords', 'training_export_raw')

# transformation = CRF(2, .75)
transformation = MorphologicalFilter(1.0)

if not os.path.exists('training_export_post_process'):
    os.mkdir('training_export_post_process')
export_data('raw_training_predictions', 'Training', 'training_tfrecords', 'training_export_post_process', transformation)


if not os.path.exists('testing_export_raw'):
    os.mkdir('testing_export_raw')
export_data('raw_testing_predictions', 'Testing', 'testing_tfrecords', 'testing_export_raw')

# transformation = CRF(2, None)
transformation = MorphologicalFilter(1.0)

if not os.path.exists('testing_export_post_process'):
    os.mkdir('testing_export_post_process')
export_data('raw_testing_predictions', 'Testing', 'testing_tfrecords', 'testing_export_post_process', transformation)
