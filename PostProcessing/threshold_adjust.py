import numpy as np
import os
from PostProcessing.Utils import read_prediction_file, get_original_image
from trainer.volume_metrics import dc, hd, assd


def adjust_training_data(threshold):
    prediction_dir = 'DataFiles/raw_training_predictions'
    image_dir = 'DataFiles/training_tfrecords'
    metrics = {outer_key: {inner_key: [] for inner_key in ['pre_crf', 'post_crf']} for outer_key in ['dc', 'hd', 'assd']}

    for file_path in os.listdir(prediction_dir):
        name, prediction, probability = read_prediction_file(os.path.join(prediction_dir, file_path))
        image, ground_truth = get_original_image(os.path.join(image_dir, name+'.tfrecord'), True)

        metrics['dc']['pre_crf'].append(dc(prediction, ground_truth))
        metrics['hd']['pre_crf'].append(hd(prediction, ground_truth))
        metrics['assd']['pre_crf'].append(assd(prediction, ground_truth))

        crf_prediction = np.round(np.array(probability) + .5 - threshold)

        metrics['dc']['post_crf'].append(dc(crf_prediction, ground_truth))
        metrics['hd']['post_crf'].append(hd(crf_prediction, ground_truth))
        metrics['assd']['post_crf'].append(assd(crf_prediction, ground_truth))

    return metrics


def report_metric(pre_crf, post_crf):
    for name, fn in zip(['Mean', 'Standard Deviation', 'Maximum', 'Minimum'], [np.mean, np.std, np.max, np.min]):
        pre = fn(pre_crf)
        post = fn(post_crf)
        print('\t{0}'.format(name))
        print('\t\tpre crf: {0:.3f} \t post crf {1:.3f} \t change: {2:.3f}%'.format(pre, post, (post-pre)/pre*100))

os.chdir('../')
thresholds = [.6, .7]

output = []
for threshold in thresholds:
    print(threshold)
    m = adjust_training_data(threshold)
    for key, metric in m.iteritems():
        names = {'dc': 'Dice Coefficient', 'hd': 'Hausdorff Distance', 'assd': 'Average Symmetric Surface Distance'}
        print(' ')
        print(names[key])
        report_metric(metric['pre_crf'], metric['post_crf'])

