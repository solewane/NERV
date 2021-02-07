'''
from torchvision import transforms
import torch
import torch.nn as nn
import math
'''
import numpy as np
import random
import torchvision.transforms.functional as F
import scipy.ndimage
from scipy.ndimage.filters import gaussian_filter, gaussian_gradient_magnitude
from skimage.transform import resize
from skimage.morphology import label, ball
from skimage.morphology.binary import binary_erosion, binary_dilation, binary_closing, binary_opening
from copy import deepcopy


class RandomBinaryOperatorTransform(object):
    def __init__(self, p_per_sample=0.4, any_of_these=(binary_dilation, binary_closing), strel_size=1):
        self.strel_size = strel_size
        self.any_of_these = any_of_these
        self.p_per_sample = p_per_sample

    def __call__(self, lbl):
        if np.random.uniform() < self.p_per_sample:
            operation = np.random.choice(self.any_of_these)
            selem = ball(self.strel_size)
            lbl = operation(lbl, selem).astype(lbl.dtype)
        return lbl


class RandomHFlip(object):
    '''
    Random horizontal flip with the probability of p
    '''

    def __call__(self, img, lbl, p=0.5):
        if random.random() < p:
            return F.hflip(img), F.hflip(lbl)
        return img, lbl


class Normalize(object):
    '''
    Intensity values between min_CTNum and max_CTNum are linearly normalized into the range [0, 1],
    intensities less than the min_CTNum are set to 0 and those greater than max_CTNum are set to 1
    '''

    def __call__(self, img, min_CTnum, max_CTnum):
        img = (img - min_CTnum) / (max_CTnum - min_CTnum)
        img[img < 0] = 0
        img[img > 1] = 1
        return img

class Zoom(object):
    def __init__(self, ori_shape, tar_shape):
        tmp = tuple([np.arange(j) / (j-1) * (i-1) for i, j in zip(ori_shape, tar_shape)])
        self.coords = np.array(np.meshgrid(*tmp, indexing='ij')).astype(float)


    def __call__(self, img, order = 0):
        return scipy.ndimage.map_coordinates(img.astype(float), self.coords, order=order, mode='constant', cval=0.0).astype(
            img.dtype)



class RandomSpatialTransforms(object):
    '''
    Random rotation, while the angle by which is randomly chosen between the range [min_angle, max_angle]
    '''

    def __call__(self, img, lbl, shape,
                 is_rotate, p_rotate, rotate_range,
                 is_scale, p_scale, scale_range):
        coords = create_zero_centered_coordinate_mesh(shape)
        #  rotate
        if is_rotate and np.random.uniform() < p_rotate:
            rotate_angle_x = random.uniform(rotate_range[0], rotate_range[1]) / 180 * np.pi
            rotate_angle_y = random.uniform(rotate_range[2], rotate_range[3]) / 180 * np.pi
            rotate_angle_z = random.uniform(rotate_range[4], rotate_range[5]) / 180 * np.pi
            coords = rotate_coords_3d(coords, rotate_angle_x, rotate_angle_y, rotate_angle_z)
        if is_scale and np.random.uniform() < p_scale:
            scale = random.uniform(scale_range[0], scale_range[1])
            coords = scale_coords(coords, scale)
        for d in range(len(img.shape)):
            coords[d] += ((np.array(shape).astype(float) - 1) / 2.)[d]
        return interpolate_img(img, coords, order=3, mode='constant', cval=0.0, is_seg=False), \
               interpolate_img(lbl, coords, order=0, mode='constant', cval=0.0, is_seg=True)


# class Padding(object):
#     '''
#     3-D padding to 2^n size
#     '''
#
#     def __init__(self, power):
#         self.multiple = 2 ** power
#
#     def __call__(self, img):
#         # assert img.shape == lbl.shape, print('size of image should match that of label')
#         imgshape = img.shape
#         assert len(imgshape) == 3, print('image dimension should be 3')
#         z_padding_num = math.ceil(imgshape[0] / self.multiple) * self.multiple - imgshape[0]
#         x_padding_num = math.ceil(imgshape[1] / self.multiple) * self.multiple - imgshape[1]
#         y_padding_num = math.ceil(imgshape[2] / self.multiple) * self.multiple - imgshape[2]
#         pad = (0, y_padding_num, 0, x_padding_num, 0, z_padding_num)
#         # starting from the last dimension and moving forward
#         return nn.functional.pad(img, pad, mode='constant', value=0)


def interpolate_img(img, coords, order=3, mode='constant', cval=0.0, is_seg=False):
    if is_seg and order != 0:
        unique_labels = np.unique(img)
        result = np.zeros(coords.shape[1:], img.dtype)
        for i, c in enumerate(unique_labels):
            res_new = scipy.ndimage.map_coordinates((img == c).astype(float), coords, order=order, mode=mode, cval=cval)
            result[res_new >= 0.5] = c
        return result
    else:
        return scipy.ndimage.map_coordinates(img.astype(float), coords, order=order, mode=mode, cval=cval).astype(
            img.dtype)


def create_zero_centered_coordinate_mesh(shape):
    tmp = tuple([np.arange(i) for i in shape])
    coords = np.array(np.meshgrid(*tmp, indexing='ij')).astype(float)
    for d in range(len(shape)):
        coords[d] -= ((np.array(shape).astype(float) - 1) / 2.)[d]
    return coords


def rotate_coords_3d(coords, angle_x, angle_y, angle_z):
    rot_matrix = np.identity(len(coords))
    rot_matrix = create_matrix_rotation_x_3d(angle_x, rot_matrix)
    rot_matrix = create_matrix_rotation_y_3d(angle_y, rot_matrix)
    rot_matrix = create_matrix_rotation_z_3d(angle_z, rot_matrix)
    coords = np.dot(coords.reshape(len(coords), -1).transpose(), rot_matrix).transpose().reshape(coords.shape)
    return coords


def create_matrix_rotation_x_3d(angle, matrix=None):
    rotation_x = np.array([[1, 0, 0],
                           [0, np.cos(angle), -np.sin(angle)],
                           [0, np.sin(angle), np.cos(angle)]])
    if matrix is None:
        return rotation_x
    return np.dot(matrix, rotation_x)


def create_matrix_rotation_y_3d(angle, matrix=None):
    rotation_y = np.array([[np.cos(angle), 0, np.sin(angle)],
                           [0, 1, 0],
                           [-np.sin(angle), 0, np.cos(angle)]])
    if matrix is None:
        return rotation_y
    return np.dot(matrix, rotation_y)


def create_matrix_rotation_z_3d(angle, matrix=None):
    rotation_z = np.array([[np.cos(angle), -np.sin(angle), 0],
                           [np.sin(angle), np.cos(angle), 0],
                           [0, 0, 1]])
    if matrix is None:
        return rotation_z
    return np.dot(matrix, rotation_z)


def scale_coords(coords, scale):
    if isinstance(scale, (tuple, list, np.ndarray)):
        assert len(scale) == len(coords)
        for i in range(len(scale)):
            coords[i] *= scale[i]
    else:
        coords *= scale
    return coords


def elastic_deform_coordinates(coordinates, alpha, sigma):
    n_dim = len(coordinates)
    offsets = []
    for _ in range(n_dim):
        offsets.append(
            gaussian_filter((np.random.random(coordinates.shape[1:]) * 2 - 1), sigma, mode="constant", cval=0) * alpha)
    offsets = np.array(offsets)
    indices = offsets + coordinates
    return indices


def elastic_deform_coordinates_2(coordinates, sigmas, magnitudes):
    '''
    magnitude can be a tuple/list
    :param coordinates:
    :param sigma:
    :param magnitude:
    :return:
    '''
    if not isinstance(magnitudes, (tuple, list)):
        magnitudes = [magnitudes] * (len(coordinates) - 1)
    if not isinstance(sigmas, (tuple, list)):
        sigmas = [sigmas] * (len(coordinates) - 1)
    n_dim = len(coordinates)
    offsets = []
    for d in range(n_dim):
        random_values = np.random.random(coordinates.shape[1:]) * 2 - 1
        random_values_ = np.fft.fftn(random_values)
        deformation_field = scipy.ndimage.fourier_gaussian(random_values_, sigmas)
        deformation_field = np.fft.ifftn(deformation_field).real
        offsets.append(deformation_field)
        mx = np.max(np.abs(offsets[-1]))
        offsets[-1] = offsets[-1] / (mx / (magnitudes[d] + 1e-8))
    offsets = np.array(offsets)
    indices = offsets + coordinates
    return indices


def augment_zoom(sample_data, sample_seg, zoom_factors, order=3, order_seg=1, cval_seg=0):
    """
    zooms data (and seg) by factor zoom_factors
    :param sample_data: np.ndarray or list/tuple of np.ndarrays, must be (c, x, y(, z))) (if list/tuple then each entry
    must be of this shape!)
    :param zoom_factors: int or list/tuple of int (multiplication factor for the input size)
    :param order: interpolation order for data (see skimage.transform.resize)
    :param order_seg: interpolation order for seg (see skimage.transform.resize)
    :param cval_seg: cval for segmentation (see skimage.transform.resize)
    :param sample_seg: can be None, if not None then it will also be zoomed by zoom_factors. Can also be list/tuple of
    np.ndarray (just like data). Must also be (c, x, y(, z))
    :return:
    """

    dimensionality = len(sample_data.shape) - 1
    shape = np.array(sample_data.shape[1:])
    if not isinstance(zoom_factors, (list, tuple)):
        zoom_factors_here = np.array([zoom_factors] * dimensionality)
    else:
        assert len(zoom_factors) == dimensionality, "If you give a tuple/list as target size, make sure it has " \
                                                    "the same dimensionality as data!"
        zoom_factors_here = np.array(zoom_factors)
    target_shape_here = list(np.round(shape * zoom_factors_here).astype(int))

    sample_data = resize_multichannel_image(sample_data, target_shape_here, order)

    if sample_seg is not None:
        target_seg = np.ones([sample_seg.shape[0]] + target_shape_here)
        for c in range(sample_seg.shape[0]):
            target_seg[c] = resize_segmentation(sample_seg[c], target_shape_here, order_seg, cval_seg)
    else:
        target_seg = None

    return sample_data, target_seg


def resize_segmentation(segmentation, new_shape, order=3, cval=0):
    '''
    Resizes a segmentation map. Supports all orders (see skimage documentation). Will transform segmentation map to one
    hot encoding which is resized and transformed back to a segmentation map.
    This prevents interpolation artifacts ([0, 0, 2] -> [0, 1, 2])
    :param segmentation:
    :param new_shape:
    :param order:
    :return:
    '''
    tpe = segmentation.dtype
    unique_labels = np.unique(segmentation)
    assert len(segmentation.shape) == len(new_shape), "new shape must have same dimensionality as segmentation"
    if order == 0:
        return resize(segmentation.astype(float), new_shape, order, mode="constant", cval=cval, clip=True,
                      anti_aliasing=False).astype(tpe)
    else:
        reshaped = np.zeros(new_shape, dtype=segmentation.dtype)

        for i, c in enumerate(unique_labels):
            mask = segmentation == c
            reshaped_multihot = resize(mask.astype(float), new_shape, order, mode="edge", clip=True,
                                       anti_aliasing=False)
            reshaped[reshaped_multihot >= 0.5] = c
        return reshaped


def resize_multichannel_image(multichannel_image, new_shape, order=3):
    '''
    Resizes multichannel_image. Resizes each channel in c separately and fuses results back together
    :param multichannel_image: c x x x y (x z)
    :param new_shape: x x y (x z)
    :param order:
    :return:
    '''
    tpe = multichannel_image.dtype
    new_shp = [multichannel_image.shape[0]] + list(new_shape)
    result = np.zeros(new_shp, dtype=multichannel_image.dtype)
    for i in range(multichannel_image.shape[0]):
        result[i] = resize(multichannel_image[i].astype(float), new_shape, order, "constant", 0, True,
                           anti_aliasing=False)
    return result.astype(tpe)


def augment_zoom(sample_data, sample_seg, zoom_factors, order=3, order_seg=1, cval_seg=0):
    """
    zooms data (and seg) by factor zoom_factors
    :param sample_data: np.ndarray or list/tuple of np.ndarrays, must be (c, x, y(, z))) (if list/tuple then each entry
    must be of this shape!)
    :param zoom_factors: int or list/tuple of int (multiplication factor for the input size)
    :param order: interpolation order for data (see skimage.transform.resize)
    :param order_seg: interpolation order for seg (see skimage.transform.resize)
    :param cval_seg: cval for segmentation (see skimage.transform.resize)
    :param sample_seg: can be None, if not None then it will also be zoomed by zoom_factors. Can also be list/tuple of
    np.ndarray (just like data). Must also be (c, x, y(, z))
    :return:
    """

    dimensionality = len(sample_data.shape) - 1
    shape = np.array(sample_data.shape[1:])
    if not isinstance(zoom_factors, (list, tuple)):
        zoom_factors_here = np.array([zoom_factors] * dimensionality)
    else:
        assert len(zoom_factors) == dimensionality, "If you give a tuple/list as target size, make sure it has " \
                                                    "the same dimensionality as data!"
        zoom_factors_here = np.array(zoom_factors)
    target_shape_here = list(np.round(shape * zoom_factors_here).astype(int))

    sample_data = resize_multichannel_image(sample_data, target_shape_here, order)

    if sample_seg is not None:
        target_seg = np.ones([sample_seg.shape[0]] + target_shape_here)
        for c in range(sample_seg.shape[0]):
            target_seg[c] = resize_segmentation(sample_seg[c], target_shape_here, order_seg, cval_seg)
    else:
        target_seg = None

    return sample_data, target_seg
