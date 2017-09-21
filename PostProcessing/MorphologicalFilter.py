import numpy as np
from scipy import ndimage
import os
from PostProcessing.Utils import adjust_training_data


def remove(prediction, threshold=.5):
    new_prediction = prediction
    labeled, n_objects = ndimage.label(prediction > 0)
    max_area = 0
    for object in range(1, n_objects + 1):
        area = np.sum(prediction[labeled == object])
        if area > max_area:
            max_area = area
    for object in range(1, n_objects + 1):
        area = np.sum(prediction[labeled == object])
        if area < threshold * max_area:
            new_prediction[labeled == object] = 0
    return new_prediction


os.chdir('../')


def transform(prediction, probability, image):
    return remove(prediction, .75)

m = adjust_training_data(transform)


