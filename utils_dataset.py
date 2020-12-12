from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
import random
import torchvision.transforms.functional as F
import torch
import torch.nn as nn
import math
import scipy.ndimage


class CoarseResolutionDataset(Dataset):
    def __init__(self,
                 pair_list,
                 padding_power=4,
                 is_train=True,
                 is_supervision=True,
                 min_ctnum=-1000,
                 max_ctnum=2200,
                 min_rotate_angle=-10,
                 max_rotate_angle=10):
        super(CoarseResolutionDataset, self).__init__()
        self.pair_list = pair_list
        self.is_train = is_train
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

    def __getitem__(self, item):
        img = torch.from_numpy(np.load(self.pair_list[item][0]).astype(np.float32))
        lbl = torch.from_numpy(np.load(self.pair_list[item][1]))
        img = img[:128, :128, :128]
        lbl = lbl[:128, :128, :128]
        img = self.Normalize(img, self.min_ctnum, self.max_ctnum)
        if self.is_train:
            img, lbl = self.RandomRotate(img, lbl, self.min_rotate_angle, self.max_rotate_angle)
            img, lbl = self.RandomHFlip(img, lbl)
        img = self.Padding(img)
        lbl = self.Padding(lbl)
        img = img * 2 - 1  # [0,1] -> [-1,1]
        lbl_downsample2 = scipy.ndimage.interpolation.zoom(lbl, 1/2, mode='nearest', order=0)
        lbl_downsample4 = scipy.ndimage.interpolation.zoom(lbl, 1/4, mode='nearest', order=0)
        lbl_downsample8 = scipy.ndimage.interpolation.zoom(lbl, 1/8, mode='nearest', order=0)
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
        for iter_, (x, y) in enumerate(test_dataloader):
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
