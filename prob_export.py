from PostProcessing.Utils import  export_data, export_data_prob
import os

#export_dir = 'temp/test_label'
#if not os.path.exists(export_dir):
#    os.mkdir(export_dir)
#export_data('DataFiles/raw_testing_predictions', 'DataFiles/Testing', 'DataFiles/testing_tfrecords', export_dir)


export_dir = 'temp/test_prob'
if not os.path.exists(export_dir):
    os.mkdir(export_dir)
export_data_prob('DataFiles/raw_testing_predictions', 'DataFiles/Testing', export_dir)