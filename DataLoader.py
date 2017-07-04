import numpy as np
import pickle
import nibabel as nib
import os


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
    return np.stack(image_list).transpose((1, 2, 3, 0))


# 4DPWI and ADC are raw data, the others are derived maps, OT is the expert segmentation
# 4DPWI has a time dimension of 80 seconds (and already has the channel dimension added in)
image_types = ['.MR_4DPWI', '.MR_ADC', '.MR_MTT', '.MR_rCBF', '.MR_rCBV', '.MR_Tmax', '.MR_TTP', '.OT']
modes_to_use = ['.MR_ADC', '.MR_MTT', '.MR_rCBF', '.MR_rCBV']

data = {key: [] for key in ['images', 'ground_truth']}


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
            data['ground_truth'].append(image)
        if image_type in modes_to_use:
            buffer.append(contrast_normalization(image))

    data['images'].append(build_multimodal_image(buffer))



# get sizes of data if they were to be save to file

for key in data:
    size = 0
    for el in data[key]:
        size += get_byte_size_of_memmap(el)
    print('File size of type {0} would be {1:.1f} MB.'.format(key, size/1e6))


for key in data:
    filename = os.path.join('ProcessedData', 'BrainArray.' + key + '.pickle')
    with open(filename, 'wb') as f:
        print('Pickling file: {0}'.format(filename))
        pickle.dump(data[key], f)
        statinfo = os.stat(filename)
        print('Compressed pickle size:', statinfo.st_size / 1e6, ' MB')

