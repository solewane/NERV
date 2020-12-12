from pathlib import Path
from utils_images import Resample, Dilate3DUsingCubicKernel
from utils import GetDataEDAfromCSV, GetOrignalPixels, GetLabelMask
import numpy as np
import time

def GenerateCoarseImage(csv_path, new_spacing=1., save_name = '1mm.npy'):
    '''
    generate coarse image and save

    Args:
        csv_path: input csv_path
        new_spacing: coarse resolution
        save_name: xxx.npy
    '''
    info_list = GetDataEDAfromCSV(csv_path)
    start_time = time.time()
    for info in info_list:
        img, ori_spacing = GetOrignalPixels(info)
        img, real_new_spacing = Resample(img, ori_spacing, new_spacing=new_spacing, interp_order=3)
        save_path = Path(info['Path']).joinpath(save_name)
        np.save(save_path, img)
        print('index:{}, cost time:{:.2f}s'.format(info['Index'], time.time() - start_time))
        print('resampled shape:{}, size:{:.2f} MB'.format(img.shape, save_path.stat().st_size/1e6))


def GenerateCoarseMask(csv_path, new_spacing=1., save_name='1mm_mask.npy', do_dilate=True):
    '''
    generate coarse mask and save

    Args:
        csv_path: input csv_path
        new_spacing: coarse resolution
        save_name: xxx.npy

    Return:
        labeled_num: labeled voxel number of mask
        total_num: total voxel number of mask
    '''
    info_list = GetDataEDAfromCSV(csv_path)
    start_time = time.time()
    labeled_num = []
    total_num = []
    for info in info_list:
        ori_shape = (int(info['Depths']), int(info['Rows']), int(info['Columns']))
        ori_spacing = float(info['Spacing'])
        mask = GetLabelMask(info, ori_shape)
        # mask = Dilate3D(image=mask, ori_spacing=ori_spacing, new_resolution=new_spacing)
        # time consuption:  resample -> dilate : 0.02s, dilate -> resample: 13.91s
        mask, _ = Resample(image=mask, ori_spacing=ori_spacing, new_spacing=new_spacing, interp_order=0)
        if do_dilate:
            mask = Dilate3DUsingCubicKernel(image=mask, ori_spacing=new_spacing, new_resolution=new_spacing)
        save_path = Path(info['Path']).joinpath(save_name)
        np.save(save_path, mask)
        print('index:{}, cost time:{:.2f}s'.format(info['Index'], time.time() - start_time))
        print('resampled shape:{}, size:{:.2f} MB'.format(mask.shape, save_path.stat().st_size/1e6))
        v = mask.sum()
        labeled_num.append(v)
        total_num.append(mask.shape[0]*mask.shape[1]*mask.shape[2])
        print('volumn cubic: {}'.format(v))
    return labeled_num, total_num





if __name__ == '__main__':
    csv_path = Path(r'data.csv')
    new_spacing = 1.
    save_name = '1mm.npy'

    # GenerateCoarseImage(csv_path=csv_path, new_spacing=new_spacing, save_name=save_name)
    a, b = GenerateCoarseMask(csv_path, save_name='1mm_mask_no_dilate.npy', do_dilate=False)