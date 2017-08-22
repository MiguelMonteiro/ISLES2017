from PostProcessing.Utils import CRF, export_data
import os

os.chdir('DataFiles/')

if not os.path.exists('training_export_raw'):
    os.mkdir('training_export_raw')
export_data('raw_training_predictions', 'Training', 'training_tfrecords', 'training_export_raw')

crf = CRF(2, .75)

if not os.path.exists('training_export_crf'):
    os.mkdir('training_export_crf')
export_data('raw_training_predictions', 'Training', 'training_tfrecords', 'training_export_crf', crf)
