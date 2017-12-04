import numpy as np
import nibabel as nib
import os
from collections import Counter


def get_sub_folders(folder):
    return [sub_folder for sub_folder in os.listdir(folder) if os.path.isdir(os.path.join(folder, sub_folder))]


def get_image_type_from_folder_name(folder_name):
    image_types = ['.MR_4DPWI', '.MR_ADC', '.MR_MTT', '.MR_rCBF', '.MR_rCBV', '.MR_Tmax', '.MR_TTP', '.OT']
    return next(image_type for image_type in image_types if image_type in folder_name)


def get_extension(filename):
    filename, extension = os.path.splitext(filename)
    return extension


def analyse_data(input_dir):

    shapes = []
    relative_volumes = []
    for folder in get_sub_folders(input_dir):
        print(folder)
        for sub_folder in get_sub_folders(os.path.join(input_dir, folder)):

            image_type = get_image_type_from_folder_name(sub_folder)

            # do not save the raw data (too heavy)
            if image_type != '.OT':
                continue

            path = os.path.join(input_dir, folder, sub_folder)
            filename = next(filename for filename in os.listdir(path) if get_extension(filename) == '.nii')
            path = os.path.join(path, filename)
            im = nib.load(path)
            image = im.get_data()
            shape = image.shape
            shapes.append(shape)
            relative_volumes.append(100 * np.sum(image) / np.cumprod(shape)[-1])
    return shapes, relative_volumes

# train
shapes, relative_volumes = analyse_data('DataFiles/Training')

print(Counter(shapes))
print('{0}+-{1}'.format(np.mean(relative_volumes), np.std(relative_volumes)))
