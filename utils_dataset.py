from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
import random
import torchvision.transforms.functional as F
import torch
import torch.nn as nn
import math
import scipy.ndimage
from skimage.transform import resize


class CoarseResolutionDataset(Dataset):
    def __init__(self,
                 pair_list,
                 padding_power=4,
                 is_train=True,
                 crop_size=(128, 128, 128),
                 min_ctnum=-1000,
                 max_ctnum=2200,
                 min_rotate_angle=-10,
                 max_rotate_angle=10):
        super(CoarseResolutionDataset, self).__init__()
        self.pair_list = pair_list
        self.is_train = is_train
        self.crop_size = crop_size
        self.Normalize = Normalize()
        self.RandomRotate = RandomRotate()
        self.RandomHFlip = RandomHFlip()
        self.min_ctnum = min_ctnum
        self.max_ctnum = max_ctnum
        self.min_rotate_angle = min_rotate_angle
        self.max_rotate_angle = max_rotate_angle
        self.Padding = Padding(padding_power)

    def __len__(self):
        return len(self.pair_list)

    def CropPadding3D(self, img, lbl, padding_value=0):
        assert img.shape == lbl.shape, print('THE SHAPE OF LABEL MUST MATCH THAT OF IMAGE!')
        out_size = np.array(self.crop_size)
        in_size = img.shape
        z_pad = -(in_size[0] - out_size[0])
        x_pad = -(in_size[1] - out_size[1])
        y_pad = -(in_size[2] - out_size[2])
        z_start = np.random.randint(0, max(-z_pad, 0) + 1)
        z_end = min(z_start + out_size[0], in_size[0])
        x_start = np.random.randint(0, max(-x_pad, 0) + 1)
        x_end = min(x_start + out_size[1], in_size[1])
        y_start = np.random.randint(0, max(-y_pad, 0) + 1)
        y_end = min(y_start + out_size[2], in_size[2])
        img = img[z_start:z_end, x_start:x_end, y_start:y_end]
        lbl = lbl[z_start:z_end, x_start:x_end, y_start:y_end]
        pad = (max(y_pad // 2, 0), max(y_pad // 2 + y_pad % 2, 0),
               max(x_pad // 2, 0), max(x_pad // 2 + x_pad % 2, 0),
               max(z_pad // 2, 0), max(z_pad // 2 + z_pad % 2, 0))
        return nn.functional.pad(img, pad, mode='constant', value=padding_value), \
               nn.functional.pad(lbl, pad, mode='constant', value=padding_value)

    def __getitem__(self, item):
        img = torch.from_numpy(np.load(self.pair_list[item][0]).astype(np.float32))
        lbl = torch.from_numpy(np.load(self.pair_list[item][1]))
        img = self.Normalize(img, self.min_ctnum, self.max_ctnum)
        if self.is_train:
            img, lbl = self.RandomRotate(img, lbl, self.min_rotate_angle, self.max_rotate_angle)
            img, lbl = self.RandomHFlip(img, lbl)
        # img = self.Padding(img)
        # lbl = self.Padding(lbl)
        print(img.shape)
        img, lbl = self.CropPadding3D(img, lbl)
        img = img * 2 - 1  # [0,1] -> [-1,1]
        lbl_downsample2 = scipy.ndimage.interpolation.zoom(lbl, 1 / 2, mode='nearest', order=0)
        lbl_downsample4 = scipy.ndimage.interpolation.zoom(lbl, 1 / 4, mode='nearest', order=0)
        lbl_downsample8 = scipy.ndimage.interpolation.zoom(lbl, 1 / 8, mode='nearest', order=0)
        # print(img.shape)
        return img, lbl, lbl_downsample2, lbl_downsample4, lbl_downsample8


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


class RandomRotate(object):
    '''
    Random rotation, while the angle by which is randomly chosen between the range [min_angle, max_angle]
    '''

    def __call__(self, img, lbl, min_angle, max_angle):
        rotate_angle = random.uniform(min_angle, max_angle)
        return F.rotate(img, rotate_angle), F.rotate(lbl, rotate_angle)


class Padding(object):
    '''
    3-D padding to 2^n size
    '''

    def __init__(self, power):
        self.multiple = 2 ** power

    def __call__(self, img):
        # assert img.shape == lbl.shape, print('size of image should match that of label')
        imgshape = img.shape
        assert len(imgshape) == 3, print('image dimension should be 3')
        z_padding_num = math.ceil(imgshape[0] / self.multiple) * self.multiple - imgshape[0]
        x_padding_num = math.ceil(imgshape[1] / self.multiple) * self.multiple - imgshape[1]
        y_padding_num = math.ceil(imgshape[2] / self.multiple) * self.multiple - imgshape[2]
        pad = (0, y_padding_num, 0, x_padding_num, 0, z_padding_num)
        # starting from the last dimension and moving forward
        return nn.functional.pad(img, pad, mode='constant', value=0)


def interpolate_img(img, coords, order=3, mode='nearest', cval=0.0, is_seg=False):
    if is_seg and order != 0:
        unique_labels = np.unique(img)
        result = np.zeros(coords.shape[1:], img.dtype)
        for i, c in enumerate(unique_labels):
            res_new = scipy.ndimage.map_coordinates((img == c).astype(float), coords, order=order, mode=mode, cval=cval)
            result[res_new >= 0.5] = c
        return result
    else:
        return scipy.ndimage.map_coordinates(img.astype(float), coords, order=order, mode=mode, cval=cval).astype(img.dtype)


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
        return resize(segmentation.astype(float), new_shape, order, mode="constant", cval=cval, clip=True, anti_aliasing=False).astype(tpe)
    else:
        reshaped = np.zeros(new_shape, dtype=segmentation.dtype)

        for i, c in enumerate(unique_labels):
            mask = segmentation == c
            reshaped_multihot = resize(mask.astype(float), new_shape, order, mode="edge", clip=True, anti_aliasing=False)
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
        result[i] = resize(multichannel_image[i].astype(float), new_shape, order, "constant", 0, True, anti_aliasing=False)
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


if __name__ == '__main__':
    from utils import GetDataEDAfromCSV
    from pathlib import Path
    from torch.utils.data import DataLoader
    from matplotlib import pyplot as plt

    save_csv_path = Path(r'data.csv')
    img_name = '1mm.npy'
    lbl_name = '1mm_mask_no_dilate.npy'

    a = GetDataEDAfromCSV(save_csv_path)
    # img_list = [Path(i['Path']).joinpath(img_name) for i in a]
    # lbl_list = [Path(i['Path']).joinpath(lbl_name) for i in a]
    data_list = [[Path(i['Path']).joinpath(img_name), Path(i['Path']).joinpath(lbl_name)] for i in a]
    # data_list = [data_list[1]]
    test_dataset = CoarseResolutionDataset(data_list, padding_power=0)

    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=1,
                                 shuffle=True,
                                 num_workers=1)
    for epoch in range(1):
        # plt.figure()
        for iter_, (x, y, _, _, _) in enumerate(test_dataloader):
            print(x.shape)
            # pass
            # if x.shape[-1] <81:
            #
            #     plt.subplot(121)
            #     plt.imshow(x[0,-1,:,:],'gray')
            # if x.shape[-1] >229:
            #     plt.subplot(122)
            #     plt.imshow(x[0,1,:,:],'gray')

            # print(y.sum())
            # plt.figure()
            # plt.subplot(121)
            # plt.imshow(x[0,62,:,:],'gray')
            # plt.subplot(122)
            # plt.imshow(y[0, 62, :, :], 'gray')
            # # print(y.shape)2222222222222

    print('hello')
