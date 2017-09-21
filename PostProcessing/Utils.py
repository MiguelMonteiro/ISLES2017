import os
import json
import numpy as np
import tensorflow as tf
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import create_pairwise_gaussian, create_pairwise_bilateral, unary_from_softmax
import nibabel as nib
from trainer.volume_metrics import dc, hd, assd


class CRF(object):
    def __init__(self, sdims, schan=None):
        self.sdims = sdims
        self.schan = schan

    def adjust_prediction(self, probability, image, iter=2):
        crf = dcrf.DenseCRF(np.prod(probability.shape), 2)
        # crf = dcrf.DenseCRF(np.prod(probability.shape), 1)

        binary_prob = np.stack((1 - probability, probability), axis=0)
        unary = unary_from_softmax(binary_prob)
        # unary = unary_from_softmax(np.expand_dims(probability, axis=0))
        crf.setUnaryEnergy(unary)

        # per dimension scale factors
        sdims = [self.sdims] * 3
        smooth = create_pairwise_gaussian(sdims=sdims, shape=probability.shape)
        crf.addPairwiseEnergy(smooth, compat=2)

        if self.schan:
            # per channel scale factors
            schan = [self.schan] * 6
            appearance = create_pairwise_bilateral(sdims=sdims, schan=schan, img=image, chdim=3)
            crf.addPairwiseEnergy(appearance, compat=2)

        result = crf.inference(iter)
        crf_prediction = np.argmax(result, axis=0).reshape(probability.shape).astype(np.float32)

        return crf_prediction


def export_data(prediction_dir, nii_image_dir, tfrecords_dir, export_dir, crf=None):

    for file_path in os.listdir(prediction_dir):
        name, prediction, probability = read_prediction_file(os.path.join(prediction_dir, file_path))

        if crf:
            image, ground_truth = get_original_image(os.path.join(tfrecords_dir, name + '.tfrecord'), False)
            prediction = crf.adjust_prediction(probability, image, iter=5)

        # build a .nii image
        img = nib.Nifti1Image(prediction, np.eye(4))
        img.set_data_dtype(dtype=np.uint8)

        path = os.path.join(nii_image_dir, name)

        adc_name = next(l for l in os.listdir(path) if 'MR_ADC' in l)
        export_image = nib.load(os.path.join(nii_image_dir, name, adc_name, adc_name + '.nii'))

        i = export_image.get_data()
        i[:] = img.get_data()

        # set name to specification and export
        _id = next(l for l in os.listdir(path) if 'MR_MTT' in l).split('.')[-1]
        export_path = os.path.join(export_dir, 'SMIR.' + name + '.' + _id + '.nii')
        nib.save(export_image, os.path.join(export_path))


def read_prediction_file(file_path):
    with open(file_path) as json_data:
        d = json.load(json_data)
    return d['name'], np.array(d['prediction']), np.array(d['probability'])


def get_original_image(tfrecords_dir, is_training_data=False):
    record = tf.python_io.tf_record_iterator(tfrecords_dir).next()
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


def report_transform_impact(pre_transform, post_transform):
    for name, fn in zip(['Mean', 'Standard Deviation', 'Maximum', 'Minimum'], [np.mean, np.std, np.max, np.min]):
        pre = fn(pre_transform)
        post = fn(post_transform)
        print('\t{0}'.format(name))
        print('\t\tpre transform: {0:.3f} \t post transform {1:.3f} \t change: {2:.3f}%'.format(pre, post, (post-pre)/pre*100))
    return


def adjust_training_data(transform, report=True):
    prediction_dir = 'DataFiles/raw_training_predictions'
    image_dir = 'DataFiles/training_tfrecords'
    metrics = {outer_key: {inner_key: [] for inner_key in ['pre_transform', 'post_transform']}
               for outer_key in ['dc', 'hd', 'assd']}

    for file_path in os.listdir(prediction_dir):
        name, prediction, probability = read_prediction_file(os.path.join(prediction_dir, file_path))
        image, ground_truth = get_original_image(os.path.join(image_dir, name+'.tfrecord'), True)

        metrics['dc']['pre_transform'].append(dc(prediction, ground_truth))
        metrics['hd']['pre_transform'].append(hd(prediction, ground_truth))
        metrics['assd']['pre_transform'].append(assd(prediction, ground_truth))

        new_prediction = transform(prediction, probability, image)

        metrics['dc']['post_transform'].append(dc(new_prediction, ground_truth))
        metrics['hd']['post_transform'].append(hd(new_prediction, ground_truth))
        metrics['assd']['post_transform'].append(assd(new_prediction, ground_truth))

    if report:
        for key, metric in metrics.iteritems():
            names = {'dc': 'Dice Coefficient', 'hd': 'Hausdorff Distance', 'assd': 'Average Symmetric Surface Distance'}
            print(' ')
            print(names[key])
            report_transform_impact(metric['pre_transform'], metric['post_transform'])

    return metrics

