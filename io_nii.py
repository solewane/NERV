import SimpleITK as sitk
import numpy as np
from matplotlib import pyplot as plt
from utils_transforms import Zoom
##  for itksnap

if __name__ == '__main__':
    testpath = r'/media/ubuntu/Seagate Backup Plus Drive/UEG/projects/radiographic_platform_00/mandible_nerve/data/excluded_data/1411048492g_air+1000/20080418/2.16.840.114421.270.9261816491.9293352491/Untitled.nii.gz'

    pred_path = r'/media/ubuntu/Seagate Backup Plus Drive/UEG/projects/radiographic_platform_00/mandible_nerve/data/original-data/nerve-7g-10/nerve-1/1900995005g/20080710/2.16.840.114421.270.9269014327.9300550327/label_pred.npy'
    lbl_path = r'/media/ubuntu/Seagate Backup Plus Drive/UEG/projects/radiographic_platform_00/mandible_nerve/data/original-data/nerve-7g-10/nerve-1/1900995005g/20080710/2.16.840.114421.270.9269014327.9300550327/1mm_mask.npy'
    save_name_lbl = r'/media/ubuntu/Seagate Backup Plus Drive/UEG/projects/radiographic_platform_00/mandible_nerve/data/original-data/nerve-7g-10/nerve-1/1900995005g/20080710/2.16.840.114421.270.9269014327.9300550327/1mm_mask.nii.gz'
    save_name_pred = r'/media/ubuntu/Seagate Backup Plus Drive/UEG/projects/radiographic_platform_00/mandible_nerve/data/original-data/nerve-7g-10/nerve-1/1900995005g/20080710/2.16.840.114421.270.9269014327.9300550327/label_pred_1mm_mask.nii.gz'

    save_name = r'/media/ubuntu/Seagate Backup Plus Drive/UEG/projects/radiographic_platform_00/mandible_nerve/data/original-data/nerve-7g-10/nerve-1/1900995005g/20080710/2.16.840.114421.270.9269014327.9300550327/visualization.nii.gz'

    depth = 576
    width = 768
    height = 768
    ori_spacing = 0.3
    new_spacing = 1.
    pred = np.load(pred_path)
    pred[pred > 0.5] = 1.
    pred[pred <= 0.5] = 0
    lbl = np.load(lbl_path)
    Z = Zoom(lbl.shape, (depth, width, height))
    a = Z(lbl)
    pred_save = Z(pred)
    savedImg = sitk.GetImageFromArray(a)
    sitk.WriteImage(savedImg, save_name_lbl)
    savedImg = sitk.GetImageFromArray(pred_save)
    sitk.WriteImage(savedImg, save_name_pred)
    c = np.zeros(a.shape)
    intersection = a * pred_save
    c[a==1] = 1
    c[pred_save==1] = 2
    c[intersection==1] = 3
    savedImg = sitk.GetImageFromArray(c)
    sitk.WriteImage(savedImg, save_name)

    # savedImg = sitk.GetImageFromArray(ct_array)
    # savedImg.SetOrigin(origin)
    # savedImg.SetDirection(direction)
    # savedImg.SetSpacing(space)
    # sitk.WriteImage(savedImg, saved_name)

    # ct = sitk.ReadImage(testpath)
    # a = sitk.GetArrayFromImage(ct)
    # idx = np.nonzero(a.sum(1).sum(1))


