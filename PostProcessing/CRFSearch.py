import os
from PostProcessing.Utils import CRF, adjust_training_data

os.chdir('../')
prediction_dir = 'DataFiles/raw_training_predictions'
image_dir = 'DataFiles/training_tfrecords'
# model search for CRF
sdims_values = [50]
schan_values = [None]
output = []
for sdims in sdims_values:
    for schan in schan_values:
        print('sdims: {0}, schan: {1}'.format(str(sdims), str(schan)))
        crf = CRF(sdims, None)
        metrics = adjust_training_data(crf, prediction_dir, image_dir)
        output.append([(sdims, schan), metrics])
