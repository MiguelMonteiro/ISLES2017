import numpy as np
import nibabel as nib
import os
import tensorflow as tf


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def to_tf_record(image_list, ground_truth_list, tf_record_file_path):

    writer = tf.python_io.TFRecordWriter(tf_record_file_path)

    for image, ground_truth in zip(image_list, ground_truth_list):

        shape = np.array(image.shape).astype(np.int32)

        # set precision of string printing to be float32
        np.set_printoptions(precision=32)

        example = tf.train.Example(features=tf.train.Features(feature={
            'shape': _bytes_feature(shape.tostring()),
            'img_raw': _bytes_feature(image.tostring()),
            'gt_raw': _bytes_feature(ground_truth.tostring())}))

        writer.write(example.SerializeToString())

    writer.close()
    print('Final file size:', os.stat(tf_record_file_path).st_size / 1e6, ' MB')


def check_if_reconstructed_images_match_originals(image_list, ground_truth_list, tf_record_file_path):

    record_iterator = tf.python_io.tf_record_iterator(path=tf_record_file_path)

    for string_record, original_image, original_ground_truth in zip(record_iterator, image_list, ground_truth_list):
        example = tf.train.Example()
        example.ParseFromString(string_record)

        shape = np.fromstring(example.features.feature['shape'].bytes_list.value[0], dtype=np.int32)
        image = np.fromstring(example.features.feature['img_raw'].bytes_list.value[0], dtype=np.float32)
        ground_truth = np.fromstring(example.features.feature['gt_raw'].bytes_list.value[0], dtype=np.uint8)

        # reshape
        img_shape = shape.tolist()
        image = image.reshape(img_shape)
        gt_shape = img_shape[:-1]
        ground_truth = ground_truth.reshape(gt_shape)

        assert np.allclose(image, original_image)
        assert np.allclose(ground_truth, original_ground_truth)


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

image_list = []
ground_truth_list = []
shapes = []
for folder in get_sub_folders('Training'):
    print(folder)
    buffer = []
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
            ground_truth_list.append(image.astype(np.uint8))
        if image_type in modes_to_use:
            buffer.append(contrast_normalization(image))

    image_list.append(build_multimodal_image(buffer))

filename = os.path.join('isles2017' + '.tfrecord')

np.unique(shapes, axis=0, return_counts=True)

to_tf_record(image_list, ground_truth_list, filename)
check_if_reconstructed_images_match_originals(image_list, ground_truth_list, filename)
