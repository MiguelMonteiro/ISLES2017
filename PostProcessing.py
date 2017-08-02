import os
import json
import numpy as np
import tensorflow as tf
from trainer.volume_metrics import dc, hd, assd
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import create_pairwise_gaussian, create_pairwise_bilateral, unary_from_softmax


def adjust_with_crf(probability, image):

    crf = dcrf.DenseCRF(np.prod(probability.shape), 2)

    binary_prob = np.stack((probability, 1 - probability), axis=-1)
    unary = unary_from_softmax(binary_prob.transpose())
    crf.setUnaryEnergy(unary)

    # per dimension scale factors
    sdims = [1] * 3
    # per channel scale factors
    schan = [1] * 6

    per_channel_scale_factor = 1
    smooth = create_pairwise_gaussian(sdims=sdims, shape=probability.shape)
    appearance = create_pairwise_bilateral(sdims=sdims, schan=schan, img=image, chdim=3)
    crf.addPairwiseEnergy(smooth, compat=2)
    crf.addPairwiseEnergy(appearance, compat=2)

    # 5 iterations
    result = crf.inference(1)

    return np.argmax(result, axis=0).reshape(probability.shape)


def read_prediction_file(file_path):
    with open(file_path) as json_data:
        d = json.load(json_data)
    return d['name'], np.array(d['prediction']), np.array(d['probability'])


def get_original_image(image_path, is_training_data=False):
    record = tf.python_io.tf_record_iterator(image_path).next()
    example = tf.train.Example()
    example.ParseFromString(record)

    shape = np.fromstring(example.features.feature['shape'].bytes_list.value[0], dtype=np.int32)
    image = np.fromstring(example.features.feature['img_raw'].bytes_list.value[0], dtype=np.float32)
    image = image.reshape(shape)

    if is_training_data:
        ground_truth = np.fromstring(example.features.feature['gt_raw'].bytes_list.value[0], dtype=np.uint8)
        ground_truth = ground_truth.reshape(shape[:-1])
    else:
        ground_truth = None

    return image, ground_truth


def adjust_training_data():
    predictions_dir = 'train_predictions'
    image_dir = 'isles_tfrecords'

    metrics = {key: [] for key in['dc_pre_crf', 'hd_pre_crf', 'assd_pre_crf', 'dc_post_crf', 'hd_post_crf',
                                  'assd_post_crf']}

    for file_path in os.listdir(predictions_dir):
        name, prediction, probability = read_prediction_file(os.path.join(predictions_dir, file_path))
        image, ground_truth = get_original_image(os.path.join(image_dir, name+'.tfrecord'), True)
        print(name)

        metrics['dc_pre_crf'].append(dc(prediction, ground_truth))
        metrics['hd_pre_crf'].append(hd(prediction, ground_truth))
        metrics['assd_pre_crf'].append(assd(prediction, ground_truth))

        crf_prediction = adjust_with_crf(probability, image)

        metrics['dc_post_crf'].append(dc(crf_prediction, ground_truth))
        metrics['hd_post_crf'].append(hd(crf_prediction, ground_truth))
        metrics['assd_post_crf'].append(assd(crf_prediction, ground_truth))
    return metrics

m = adjust_training_data()
for key, metric in m.iteritems():
    mean = np.mean(metric)
    std = np.std(metric)
    minimum = np.min(metric)
    maximum = np.max(metric)
    print(key)
    print('\tMean: {0:.2f}'.format(mean))
    print('\tStandard Deviation: {0:.2f}'.format(std))
    print('\tMinimum: {0:.2f}'.format(minimum))
    print('\tMaximum: {0:.2f}'.format(maximum))



