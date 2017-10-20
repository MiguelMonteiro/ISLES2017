from PostProcessing.Utils import MorphologicalFilter, export_data
import os

os.chdir('../DataFiles/')

transformation = MorphologicalFilter(.5)

if not os.path.exists('cv_export'):
    os.mkdir('cv_export')
for fold in range(5):
    prediction_dir = os.path.join('cv_predictions', 'fold_' + str(fold + 1))
    export_data(prediction_dir, 'Training', 'training_tfrecords', 'cv_export', transformation)
