# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 14:48:45 2020

@author: zhang
"""
import pydicom
import os
import numpy as np
#import seaborn
from matplotlib import pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
# import SimpleITK as sitk
import csv
from pathlib import Path
from typing import (Callable, Dict, List, Tuple)
import pylibjpeg
#
#example_dcm = r'F:\UEG\projects\radiographic_platform_00\mandible_nerve\data\original-data\nerve-1g-8\0007g\20120320\2.16.840.114421.11369.9385564476.9417100476\0004.dcm'
#example_dcm2 = r'F:\UEG\projects\radiographic_platform_00\mandible_nerve\data\original-data\nerve-1g-8\0007g\20120320\2.16.840.114421.11369.9385564476.9417100476\0005.dcm'
#
#dcm = pydicom.read_file(example_dcm)
#dcm2 = pydicom.read_file(example_dcm2)


def DatasetBuild(path):
    '''
    walk through path,return dcm path list
    as:     [['xxx/a.dcm','xxx/b.dcm'...],
             ['yyy/a.dcm','yyy/b.dcm'...],...]
    '''
#    dcm_dataset_file = []
    idx = 0
    for (root, dirs, files) in os.walk(path):
        dcm_file_list = [os.path.join(root, f) for f in files if f.endswith('.dcm')]
        dcm_num = len(dcm_file_list)
        if dcm_num > 1: # only 1 dcm file?
            label_file = os.path.join(root, 'LoadStatus', 'Workup000', 'ncData.bin')
            if os.path.exists(label_file):
                '''
                not all dcmfiles have corresponding label data 'ncData.bin'
                '''
#                dcm_dataset_file.append(dcm_file_list)
                dcm_tag = pydicom.read_file(dcm_file_list[0])
                spacex, spacey = dcm_tag.PixelSpacing
                voxel_size = float(spacex)
                img_height = dcm_tag.Rows
                save_name = str(idx) + '_' + str(voxel_size) + '_' + str(dcm_num) + 'x' + str(img_height) + 'x' + str(img_height) + '.npy'
                data_ = np.zeros([2, dcm_num, img_height, img_height])
                for z_idx in range(dcm_num):
                    img_2D = DicomReader(dcm_file_list[z_idx])
                    data_[0, z_idx, :, :] = img_2D
                label_coor = GetCoor(label_file)#x,y,z
                data_[1,label_coor[2,:],label_coor[0,:],label_coor[1,:]] = 1
                np.save(save_name, data_)
                del data_
                print(save_name + ' save successfully')
                idx += 1
#    print(num)
#    return dcm_dataset_file,num


def CheckDcmInfo(path):
    '''
    walk through path,return dcm path list
    as:     [['xxx/a.dcm','xxx/b.dcm'...],
             ['yyy/a.dcm','yyy/b.dcm'...],...]
    '''
#    num=0
    dcm_dataset_file = []
    for (root, dirs, files) in os.walk(path):
        dcm_file = [os.path.join(root, f) for f in files if f.endswith('.dcm')]
        dcm_num = len(dcm_file)
        if dcm_num > 1: # only 1 dcm file?
#            num += 1
            dcm_dataset_file.append(dcm_file)
#    print(num)
    return dcm_dataset_file


def get_pixel_info(dcm_list):
    dcm_tag_1 = pydicom.read_file(dcm_list[0])
    spacex, spacey = dcm_tag_1.PixelSpacing
    '''
    CBCT:PixelSpacing_x = PixelSpacing_y = PixelSpacing_z
    '''
# =============================================================================
#     # space x-y
#     # InstanceNumber-sort
#     SliceLocations = []
#     ImagePositon_z = []
#     for dcm in dcm_list:
#         dcm_tag = pydicom.read_file(dcm)
#         SliceLocations.append(dcm_tag.SliceLocation)
#         ImagePositon_z.append(dcm_tag.ImagePositionPatient[2])
#     SliceLocations_max =max(SliceLocations)
#     SliceLocations_min =min(SliceLocations)
#     ImagePositon_z_max = max(ImagePositon_z)
#     ImagePositon_z_min = min(ImagePositon_z)
# #    print(SliceLocations_max)
# #    print(SliceLocations_min)
# #    print(ImagePositon_z_max)
# #    print(ImagePositon_z_min)
#     if SliceLocations_max - SliceLocations_min < 1e-10:
#         spacez = abs(ImagePositon_z_max - ImagePositon_z_min)/(len(dcm_list)-1)
#     else:
#         spacez = abs(SliceLocations_max - SliceLocations_min)/(len(dcm_list)-1)
#     pixel_infos = [float(spacex), float(spacey), spacez]
# =============================================================================
    return float(spacex)


def GetCoor(mask_path):
    '''
    19:end  x1,y1,z1,x2,y2,z2,...
    '''
    b = np.fromfile(mask_path, dtype = np.int32)
    b = b[19:]
    b = b.reshape(-1, 3).transpose()
    _, idx = np.where(b == np.max(b))
    b = np.delete(b, idx, axis=1)    # OUTLIER
    _, idx = np.where(b == np.min(b))
    b = np.delete(b, idx, axis=1)    # NOVELTY
    return b


def DicomReader(dicompath):
    img = sitk.ReadImage(dicompath)
    img_array = sitk.GetArrayFromImage(img) #z, x, y
    return img_array[0, :, :] # int16 numpy


def GetDataEDA2CSV(data_path: Path, output_csv_path: Path)->None:
    '''
    data Exploratory Data Analysis, output to a csv file.

    Args:
        data_path: dicom file path
        output_csv_path: the output csv file
    '''
    dcmfolde_to_dcmnum = GetDcmFolders(data_path)
    f = open(output_csv_path, 'w', encoding='utf-8')
    csv_writer = csv.writer(f)
    head = ['Index', 'Path', 'Dicom_Num', 'Rows', 'Columns', 'Depths', 'Spacing',
            'Rowsize(mm)', 'Columnsize(mm)', 'Depthsize(mm)',
            'Window_Center', 'Window_Width', 'Manufacturer', 'Institution_Name']
    path_suffix = 'LoadStatus/Workup000/ncData.bin'
    csv_writer.writerow(head)
    idx = 0
    for folder in dcmfolde_to_dcmnum.keys():
        if Path(folder).joinpath(path_suffix).exists():
        # if (folder / 'LoadStatus').exists():# ...data cleanning
            dcmfile = pydicom.read_file(list(folder.glob('*.dcm'))[0])
            dcm_voxel_size = dcmfile.PixelSpacing[0]
            rows = dcmfile.Rows
            columns = dcmfile.Columns
            dcm_num = dcmfolde_to_dcmnum[folder]# len(list(folder.glob('*.dcm')))
            window_center = dcmfile.WindowCenter
            window_width = dcmfile.WindowWidth
            manufacturer = dcmfile.Manufacturer
            institution_name = dcmfile.InstitutionName
            if dcm_num == 1: # only 1 dcm file
                z_size = dcmfile.NumberOfFrames
            else:
                z_size = dcm_num
            csv_writer.writerow([idx, folder, dcm_num, rows, columns, z_size, dcm_voxel_size,
                                 rows * dcm_voxel_size, columns * dcm_voxel_size, z_size * dcm_voxel_size,
                                 window_center, window_width, manufacturer, institution_name])
            idx += 1
            # print(f'dcm shape:{dcmfile.Rows}x{dcmfile.Columns}x{z_size}, pixel size:{dcm_voxel_size}')
    f.close()


def GetDcmFolders(data_path: Path) -> Dict[Path, int]:
    '''
    Iterate through all the subfolders to get the pairs of dicom folders and dicom file numbers includes.

    Args:
        data_path: the input folder path

    Returns:
        a dictionary mapping subfolder path to dicom numbers
    '''
    dcmfolde_to_dcmnum = {}
    for file in data_path.rglob('*.dcm'):
        if file.parent not in dcmfolde_to_dcmnum.keys():
            dcmfolde_to_dcmnum[file.parent] = 1
        else:
            dcmfolde_to_dcmnum[file.parent] += 1
    assert len(dcmfolde_to_dcmnum) >0, print(f'NO dicom files found in {data_path}')
    return dcmfolde_to_dcmnum


def GetDataEDAfromCSV(input_csv_path: Path) ->List[Tuple]:
    '''
    Get data info from the input csv file

    Args:
        input_csv_path: the input csv file path

    Returns:
         [{'data1_info_key_1':value,'data1_info_key_2':value, ...},
          {'data2_info_key_1':value,'data2_info_key_2':value, ...},... ]
    '''
    assert input_csv_path.exists(), print(f'CSV file not exists: {input_csv_path}')
    output_list_of_dict = []
    with open(input_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        fieldnames = next(reader)
        csv_reader = csv.DictReader(f, fieldnames=fieldnames)# list of keys for the dict
        for row in csv_reader:
            d = {}
            for k, v in row.items():
                d[k] = v
            output_list_of_dict.append(d)
    return output_list_of_dict


def GetOrignalPixels(input_dict):
    '''
    Get original 3-D pixels from dicom files, which from the dictionary from the csv file

    Args:
        input_dict: input dictionary

    Returns:
         np.array, float(spacing)
    '''
    path = list(Path(input_dict['Path']).glob('*.dcm'))
    dicom_info = [pydicom.read_file(i) for i in path]
    if int(input_dict['Dicom_Num']) == 1:
        image = dicom_info[0].pixel_array
        image = image.astype(np.int16)
    else:
        instance_number = [int(i.InstanceNumber) for i in dicom_info]
        dicom_info_sorted = [i for _, i in sorted(zip(instance_number, dicom_info))]
        # instance_number = [int(pydicom.read_file(i).InstanceNumber) for i in path]
        # path_sorted = [i for _, i in sorted(zip(instance_number, path))]
        image = np.stack([i.pixel_array for i in dicom_info_sorted])
        image = image.astype(np.int16)
    print(image.shape, end="  ")
    print(len(dicom_info), end= "  ")
    # print(image.mean())
    return image, float(input_dict['Spacing'])


def GetLabelCoor(input_dict, path_suffix = 'LoadStatus/Workup000/ncData.bin'):
    '''
    Get the coordinates of label

    Args:
        input_dict: input dictionary

    Returns:
         np.array
    '''
    label_path = Path(input_dict['Path']).joinpath(path_suffix)
    assert label_path.exists(), print(f'label file not exists: {label_path}')
    return GetCoor(label_path)


def GetLabelMask(input_dict, img_shape):
    '''
    Get the mask of label from label coordinates

    Args:
        input_dict: input dictionary
        img_shape: image shape

    Returns:
         3d-mask
    '''
    mask = np.zeros(img_shape).astype(np.int16)
    b = GetLabelCoor(input_dict)
    for i in range(len(b.transpose())):
        mask[b[2, i], b[1, i], b[0, i]] = 1
    return mask


if __name__ == '__main__':
    '''
    voxel print
    '''
# =============================================================================
#     path_package = r'F:\UEG\projects\radiographic_platform_00\mandible_nerve\data\original-data'
#     dcmfile = CheckDcmInfo(path_package)
#     for i in range(len(dcmfile)):
#         voxel = get_pixel_info(dcmfile[i])
#         print(voxel)
# =============================================================================
    '''
    .bin label visualization
    '''
# =============================================================================
#     mask_path = r'/media/ubuntu/Seagate Backup Plus Drive/UEG/projects/radiographic_platform_00/mandible_nerve/kavo-data-decode/ncData.bin'
# #    f = open(mask_path, 'rb')
# #    a = f.read()
# #    print(f.name)
# #    f.close()
#     b = GetCoor(mask_path)
#     fig = plt.figure()
#
#     ax = fig.add_subplot(111, projection='3d')
# #    fig = plt.figure()
# ##    ax = plt.subplot(111, projection = '3d')
# #    ax = fig.gca(projection="3d", title="plot title")
#     ax.scatter(b[0,:], b[1,:], b[2,:])
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.set_zlabel('z')
#     plt.show()
# #
# =============================================================================
    '''
    plot dicom
    '''
    # example_dcm = r'/media/ubuntu/Seagate Backup Plus Drive/UEG/projects/radiographic_platform_00/mandible_nerve/data/original-data/nerve-6g-8/nerve-1/0520497458g/20080318/2.16.840.114421.270.9259139166.9290675166/0061.dcm'
    # dcm = pydicom.read_file(example_dcm)
    # img = DicomReader(example_dcm)
    # print('successfully')
    # plt.imshow(img, 'gray')
    # plt.show()
# =============================================================================
#     path_package = r'F:\UEG\projects\radiographic_platform_00\mandible_nerve\data\original-data'
#     DatasetBuild(path_package)
# =============================================================================


    testpath = Path(r'/media/ubuntu/Seagate Backup Plus Drive/UEG/projects/radiographic_platform_00/mandible_nerve/data/original-data')
    save_csv_path = Path(r'data.csv')
    GetDataEDA2CSV(testpath, save_csv_path)
    a = GetDataEDAfromCSV(save_csv_path)
    # b = GetOrignalPixels(a[1])

    # for c in range(len(a)-1):
    #     # dd = GetLabelCoor(a[c])
    #     # print(dd.shape)
    #     cc=GetOrignalPixels(a[c])
    #     # tmp=pydicom.read_file(cc[0])
    #     # tmp=tmp.pixel_array
    #     print(c)
    # dcm1 = Path(r'/media/ubuntu/Seagate Backup Plus Drive/UEG/projects/radiographic_platform_00/mandible_nerve/data/original-data/nerve-9/nerve-1/TJ0A000650/20140808/2.16.840.114421.11794.9460828425.9492364425/0011.dcm')
    # dcm = pydicom.read_file(dcm1)
    # p = Path(r'/home/zhangyunxu/Deployment')
    # GetDcmFolders(p)

