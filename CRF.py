import tensorflow as tf
import numpy as np

# # 2D case
# a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
#
# a = a.reshape([1, 1, 3, 3])
#
# b = np.sqrt(a * a - 2 * a * a.transpose() + a.transpose() * a.transpose())
#
# b.transpose((0, 1, 3, 2))
#
# # 3D case
#
# a = np.array(
#     [[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
#
# a = a.reshape([1, 1, 1, 3, 3, 3])
#
# b = np.sqrt(a * a - 2 * a * a.transpose() + a.transpose() * a.transpose())


# Calculates pixel/voxel pairwise distance for n-dimensional image
def calc_n_dimensional_pixel_pairwise_squared_distance(image):
    new_shape = tf.concat((tf.tile([1], [tf.rank(image)]), tf.shape(image)), axis=0)
    im = tf.reshape(image, new_shape)
    im_t = tf.transpose(im)
    return im * im - 2 * im * im_t + im_t * im_t


def crf(probs, image):
    unary = -tf.log(probs)

    # kernels
    # kernel parameters
    theta_alpha = .5
    theta_beta = .5
    theta_gamma = .5

    pairwise_class_squared_distance = calc_n_dimensional_pixel_pairwise_squared_distance(probs)

    channels = tf.unstack(image, num=6, axis=-1)

    tmp = -pairwise_class_squared_distance / (2 * tf.square(theta_alpha))
    for channel in channels:
        pairwise_voxel_squared_distance = calc_n_dimensional_pixel_pairwise_squared_distance(channel)
        tmp += -pairwise_voxel_squared_distance / (2 * tf.square(theta_beta))

    w1 = tf.Variable(initial_value=.1)
    appearance_kernel = w1 * -tf.exp(tmp)

    w2 = tf.Variable(initial_value=.1)
    smoothness_kernel = w2 * tf.exp(-tf.square(pairwise_class_squared_distance) / (2 * tf.square(theta_gamma)))

    logits = -unary - tf.reduce_sum(appearance_kernel + smoothness_kernel)
    return logits
    # E = unary + tf.reduce_sum(appearance_kernel + smoothness_kernel)
    # return 1 / (1 + E)


def extract_image_patches_3D(image, ksizes, strides, rates, padding, name=None):
    tf.strided_slice(image, [0, 0, 0, 0, 0, 0], image.shape, strides, begin_mask=, end_mask=)
    strides=None,

tf.extract_image_patches()

def crf(probs, image, patch):
    unary = -tf.log(probs)


    # kernels
    # kernel parameters
    theta_alpha = .5
    theta_beta = .5
    theta_gamma = .5

    # use a convolution with identity filter to extract a patch
    w = tf.constant(1.0, shape=patch+[6,6])
    tf.nn.conv3d(image, w, [1, 1, 1, 1, 1], padding='SAME')

    zeros = np.zeros(shape=np.prod(patch)[0])
    for i in range(np.prod(patch)[0]):
        a =  zeros
        a[i] = -1
        a = a.reshape(patch)
        w = tf.constant(1.0, shape=a + [6, 6])
        b= zeros
        b[0]=1
        a = a.reshape(patch)
        b = tf.constant(1.0, shape=a + [6, 6])

        2 * tf.nn.conv3d(image, _, [1, 1, 1, 1, 1], padding='SAME') -  tf.nn.conv3d(image, w, [1, 1, 1, 1, 1], padding='SAME')



    pairwise_class_squared_distance = calc_n_dimensional_pixel_pairwise_squared_distance(probs)

    channels = tf.unstack(image, num=6, axis=-1)

    tmp = -pairwise_class_squared_distance / (2 * tf.square(theta_alpha))
    for channel in channels:
        pairwise_voxel_squared_distance = calc_n_dimensional_pixel_pairwise_squared_distance(channel)
        tmp += -pairwise_voxel_squared_distance / (2 * tf.square(theta_beta))

    w1 = tf.Variable(initial_value=.1)
    appearance_kernel = w1 * -tf.exp(tmp)

    w2 = tf.Variable(initial_value=.1)
    smoothness_kernel = w2 * tf.exp(-tf.square(pairwise_class_squared_distance) / (2 * tf.square(theta_gamma)))

    logits = -unary - tf.reduce_sum(appearance_kernel + smoothness_kernel)
    return logits
    # E = unary + tf.reduce_sum(appearance_kernel + smoothness_kernel)
    # return 1 / (1 + E)