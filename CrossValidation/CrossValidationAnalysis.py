from PostProcessing.Utils import MorphologicalFilter, adjust_training_data, report_transform_impact
import os

os.chdir('../')
image_dir = 'DataFiles/training_tfrecords'

folds = list()
for i in range(5):
    print('Fold: {0}'.format(str(i+1)))
    mf = MorphologicalFilter(.9)
    prediction_dir = 'DataFiles/cv_predictions/fold_' + str(i+1)
    metrics = adjust_training_data(mf, prediction_dir, image_dir, report=False)
    folds.append(metrics)

metrics = {outer_key: {inner_key: [] for inner_key in ['pre_transform', 'post_transform']}
     for outer_key in ['dc', 'hd', 'assd']}


for outer_key in ['dc', 'hd', 'assd']:
    for inner_key in ['pre_transform', 'post_transform']:
        for fold in folds:
            metrics[outer_key][inner_key] += fold[outer_key][inner_key]

    names = {'dc': 'Dice Coefficient\n', 'hd': 'Hausdorff Distance\n', 'assd': 'Average Symmetric Surface Distance\n'}
    print(names[outer_key])
    report_transform_impact(metrics[outer_key]['pre_transform'], metrics[outer_key]['post_transform'])
