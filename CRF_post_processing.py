import numpy as np
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
    result = crf.inference(2)

    return np.argmax(result, axis=0).reshape(probability.shape)