import numpy as np
from scipy import ndimage
import os
from PostProcessing.Utils import adjust_training_data


def remove(prediction, threshold=.5):

    new_prediction = prediction

    labeled, n_objects = ndimage.label(prediction > 0)

    max_volume = 0

    volumes = {}

    for object_n in range(1, n_objects + 1):
        volume = np.sum(prediction[labeled == object_n])
        if volume > max_volume:
            max_volume = volume
        volumes.update({object_n: volume})

    for object_n, volume in volumes.iteritems():
        if volume < threshold * max_volume:
            new_prediction[labeled == object_n] = 0
    return new_prediction


os.chdir('../')


def transform(prediction, probability, image):
    return remove(prediction, .99)

m = adjust_training_data(transform)


