import numpy as np
import nibabel as nib
import os
import tensorflow as tf


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def to_tfrecord(data, file_dir):

    for key, values in data.iteritems():
        writer = tf.python_io.TFRecordWriter(os.path.join(file_dir, key+'.tfrecord'))
        image = values['image']
        ground_truth = values['ground_truth']

        shape = np.array(image.shape).astype(np.int32)

        # set precision of string printing to be float32
        np.set_printoptions(precision=32)

        example = tf.train.Example(features=tf.train.Features(feature={
            'example_name': _bytes_feature(key),
            'shape': _bytes_feature(shape.tostring()),
            'img_raw': _bytes_feature(image.tostring()),
            'gt_raw': _bytes_feature(ground_truth.tostring())}))

        writer.write(example.SerializeToString())
        writer.close()


def get_byte_size_of_memmap(array):
    return np.prod(array.shape) * np.dtype(array.dtype).itemsize


def get_sub_folders(folder):
    return [sub_folder for sub_folder in os.listdir(folder) if os.path.isdir(os.path.join(folder, sub_folder))]


def get_image_type_from_folder_name(folder_name):
    return next(image_type for image_type in image_types if image_type in folder_name)


def get_extension(filename):
    filename, extension = os.path.splitext(filename)
    return extension


def contrast_normalization(image, min_divisor=1e-3):
    mean = image.mean()
    std = image.std()
    if std < min_divisor:
        std = min_divisor
    return (image - mean) / std


def build_multimodal_image(image_list):
    shape = image_list[0].shape
    for image in image_list:
        assert image.shape == shape
    return np.stack(image_list).transpose((1, 2, 3, 0)).astype(np.float32)


# 4DPWI and ADC are raw data, the others are derived maps, OT is the expert segmentation
# 4DPWI has a time dimension of 80 seconds (and already has the channel dimension added in)
image_types = ['.MR_4DPWI', '.MR_ADC', '.MR_MTT', '.MR_rCBF', '.MR_rCBV', '.MR_Tmax', '.MR_TTP', '.OT']
modes_to_use = ['.MR_ADC', '.MR_MTT', '.MR_rCBF', '.MR_rCBV', '.MR_Tmax', '.MR_TTP']


shapes = []
data = {}

for folder in get_sub_folders('Training'):
    print(folder)
    buffer = []
    data[folder] = {}
    for sub_folder in get_sub_folders(os.path.join('Training', folder)):

        image_type = get_image_type_from_folder_name(sub_folder)

        # do not save the raw data (too heavy)
        if image_type == '.MR_4DPWI':
            continue

        path = os.path.join('Training', folder, sub_folder)
        filename = next(filename for filename in os.listdir(path) if get_extension(filename) == '.nii')
        path = os.path.join(path, filename)
        im = nib.load(path)
        image = im.get_data()

        if image_type == '.OT':
            shapes.append(image.shape)
            data[folder]['ground_truth'] = image.astype(np.uint8)
        if image_type in modes_to_use:
            buffer.append(contrast_normalization(image))

    data[folder]['image'] = build_multimodal_image(buffer)


np.unique(shapes, axis=0, return_counts=True)
file_dir = 'isles_tfrecords'
to_tfrecord(data, file_dir)
