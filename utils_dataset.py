'''
from torchvision import transforms
import random
import torchvision.transforms.functional as F
import math
from scipy.ndimage.filters import gaussian_filter, gaussian_gradient_magnitude
from skimage.transform import resize
'''
from torch.utils.data import Dataset
import numpy as np
import torch
import torch.nn as nn
import scipy.ndimage
from utils_transforms import RandomSpatialTransforms, RandomHFlip, Normalize, RandomBinaryOperatorTransform


class CoarseResolutionDataset(Dataset):
    def __init__(self,
                 pair_list,
                 # padding_power=4,
                 is_train=True,
                 crop_size=(128, 128, 128),
                 min_ctnum=-1000,
                 max_ctnum=2200,
                 is_croppadding=True,
                 is_flip=True,
                 is_rotate=True,
                 p_rotate=0.2,
                 rotate_range=(-10, 10, -10, 10, -10, 10),
                 is_scale=True,
                 p_scale=0.2,
                 scale_range=(0.8, 1.25),
                 is_morphology=True,
                 p_morphology=0.4,
                 ):
        super(CoarseResolutionDataset, self).__init__()
        self.pair_list = pair_list
        self.is_train = is_train
        self.crop_size = crop_size
        self.Normalize = Normalize()
        self.RandomSpatialTransforms = RandomSpatialTransforms()
        self.RandomHFlip = RandomHFlip()
        self.RandomBinaryOperatorTransform = RandomBinaryOperatorTransform(p_per_sample=p_morphology,
                                                                           strel_size=1)
        self.min_ctnum = min_ctnum
        self.max_ctnum = max_ctnum
        self.is_croppadding = is_croppadding
        self.is_flip = is_flip
        self.is_rotate = is_rotate
        self.p_rotate = p_rotate
        self.rotate_range = np.array(rotate_range)
        self.is_scale = is_scale
        self.p_scale = p_scale
        self.scale_range = np.array(scale_range)
        self.is_morphology = is_morphology
        # self.Padding = Padding(padding_power)

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
        img = np.load(self.pair_list[item][0]).astype(np.float32)
        lbl = np.load(self.pair_list[item][1])
        img = self.Normalize(img, self.min_ctnum, self.max_ctnum)
        if self.is_train:
            img, lbl = self.RandomSpatialTransforms(img, lbl, img.shape,
                                                    self.is_rotate, self.p_rotate, self.rotate_range,
                                                    self.is_scale, self.p_scale, self.scale_range)
            if self.is_morphology:
                lbl = self.RandomBinaryOperatorTransform(lbl)
            img = torch.from_numpy(img)
            lbl = torch.from_numpy(lbl)
            if self.is_flip:
                img, lbl = self.RandomHFlip(img, lbl)
        else:
            img = torch.from_numpy(img)
            lbl = torch.from_numpy(lbl)
        if self.is_croppadding:
            img, lbl = self.CropPadding3D(img, lbl)
        img = img * 2 - 1  # [0,1] -> [-1,1]
        lbl_downsample2 = scipy.ndimage.interpolation.zoom(lbl, 1 / 2, mode='nearest', order=0)
        lbl_downsample4 = scipy.ndimage.interpolation.zoom(lbl, 1 / 4, mode='nearest', order=0)
        lbl_downsample8 = scipy.ndimage.interpolation.zoom(lbl, 1 / 8, mode='nearest', order=0)
        # print(img.shape)
        return img, lbl, lbl_downsample2, lbl_downsample4, lbl_downsample8


class test(object):
    def __init__(self, a):
        self.a = a
        print('init !!!!!')

    def __call__(self, c):
        print(c + self.a)


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
    data_list = [data_list[1]]
    test_dataset = CoarseResolutionDataset(pair_list=data_list,
                                           is_train=True,
                                           crop_size=(128, 128, 128),
                                           min_ctnum=-1000,
                                           max_ctnum=2200,
                                           is_croppadding=False,
                                           is_flip=False,
                                           is_rotate=False,
                                           p_rotate=0.2,
                                           rotate_range=(-10, 10, -10, 10, -10, 10),
                                           is_scale=False,
                                           p_scale=0.2,
                                           scale_range=(0.8, 1.25),
                                           is_morphology=True,
                                           p_morphology=0.3,
                                           )

    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=1,
                                 shuffle=True,
                                 num_workers=1)
    for epoch in range(1):
        # plt.figure()
        for iter_, (x, y, _, _, _) in enumerate(test_dataloader):
            # print(x.shape)
            # print(y.shape)
            print(torch.sum(y))
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
    plt.imshow(y[0, 65, :, :], 'gray')
    print('hello')

    # m = test(2)
    # e = 3
    # m(e)
