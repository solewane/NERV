from pathlib import Path
from utils import GetDataEDAfromCSV, GetOrignalPixels, GetLabelMask
import numpy as np
import scipy.ndimage
from matplotlib import pyplot as plt
from scipy import ndimage
import time
import pydicom


def Resample(image, ori_spacing, new_spacing=1., interp_order=0):
    '''
    Resample from the original resolution to the new (isotropic) resolution
    (interpolation), the shape would be changed since

    Args:
        image: the input image
        ori_spacing: old resolution
        new_spacing: dilate kernel to the coarse resolution
        interp_order: interpolation order,
                interp_order=0(nearest interpolation) for label image(mask),
                interp_order=3(cubic spline interpolation) for dicom image

    Returns:
         resampled image, real new spacing
    '''
    # Determine current pixel spacing
    resize_factor = ori_spacing / new_spacing
    # new_real_shape = list(image.shape) * resize_factor
    new_real_shape = [i * resize_factor for i in image.shape]
    new_shape = np.round(new_real_shape)  # 返回浮点数x的四舍五入值。
    real_resize_factor = new_shape / image.shape
    real_new_spacing = ori_spacing / real_resize_factor
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest', order=interp_order)
    # when order = 0: nearest
    return image, real_new_spacing


def Dilate3D(image, ori_spacing, new_resolution):
    '''
    dilate the mask from old resolution[ori_spacing] to new resolution,
    to prevent label diminishing

    Args:
        image: the input image
        ori_spacing: old resolution
        new_resolution: dilate kernel to the coarse resolution

    Returns:
         dilated image
    '''
    dilate_radius = round(new_resolution / ori_spacing)
    morphology_struct_radius = round(dilate_radius)
    struct_side_length = morphology_struct_radius * 2 + 1
    dilate_struct = np.zeros((struct_side_length, struct_side_length, struct_side_length))
    for iz in range(struct_side_length):
        for ix in range(struct_side_length):
            for iy in range(struct_side_length):
                if (morphology_struct_radius - iz) ** 2 + \
                        (morphology_struct_radius - ix) ** 2 + \
                        (morphology_struct_radius - iy) ** 2 <= dilate_radius ** 2:
                    dilate_struct[iz, ix, iy] = 1
    output_img = ndimage.binary_dilation(image, structure=dilate_struct).astype(image.dtype)
    return output_img

def Dilate3DUsingCubicKernel(image, ori_spacing, new_resolution):
    '''
    dilate the mask from old resolution[ori_spacing] to new resolution,
    to prevent label diminishing

    Args:
        image: the input image
        ori_spacing: old resolution
        new_resolution: dilate kernel to the coarse resolution

    Returns:
         dilated image
    '''
    dilate_radius = round(new_resolution / ori_spacing)
    morphology_struct_radius = round(dilate_radius)
    struct_side_length = morphology_struct_radius * 2 + 1
    dilate_struct = np.ones((struct_side_length, struct_side_length, struct_side_length))
    output_img = ndimage.binary_dilation(image, structure=dilate_struct).astype(image.dtype)
    return output_img



if __name__ == '__main__':
    # save_csv_path = Path(r'data.csv')
    # a = GetDataEDAfromCSV(save_csv_path)
    # img_ori, ori_spacing = GetOrignalPixels(a[1])
    # b = GetLabelMask(a[1], img_ori.shape)
    # b_dilate = Dilate3D(b, ori_spacing, 1.)
    # img_3, real_new_spacing = Resample(img_ori, ori_spacing, new_spacing=1., interp_order=3)
    # print(f'real new spacing is : {real_new_spacing}')
    # img_0, real_new_spacing = Resample(b, ori_spacing, new_spacing=1., interp_order=0)
    # b_dilate, real_new_spacing = Resample(b_dilate, ori_spacing, new_spacing=1., interp_order=0)
    # plt.figure()
    # plt.subplot(231)
    # plt.imshow(img_3[69, :, :], 'gray')
    # plt.subplot(232)
    # plt.imshow(img_0[69, :, :], 'gray')
    # plt.subplot(233)
    # plt.imshow(b_dilate[69, :, :], 'gray')
    # plt.subplot(234)
    # plt.imshow(img_3[75, :, :], 'gray')
    # plt.subplot(235)
    # plt.imshow(img_0[75, :, :], 'gray')
    # plt.subplot(236)
    # plt.imshow(b_dilate[75, :, :], 'gray')
    '''
    resample -> dilate
    dilate -> resample
    '''
    save_csv_path = Path(r'data.csv')
    a = GetDataEDAfromCSV(save_csv_path)
    img_ori, ori_spacing = GetOrignalPixels(a[1])
    b = GetLabelMask(a[1], img_ori.shape)
    time1 = time.time()
    b_dilate = Dilate3D(b, ori_spacing, 1.)
    b_dilate, real_new_spacing = Resample(b_dilate, ori_spacing, new_spacing=1., interp_order=0)
    time2 = time.time()
    img_0, real_new_spacing = Resample(b, ori_spacing, new_spacing=1., interp_order=0)
    img_0 = Dilate3DUsingCubicKernel(img_0, 1., 1.)
    time3 = time.time()
    print('resample -> dilate : {:.2f}s, dilate -> resample: {:.2f}s'.format(time3-time2, time2-time1))
    plt.figure()

    plt.subplot(221)
    plt.imshow(img_0[69, :, :], 'gray')
    plt.subplot(222)
    plt.imshow(b_dilate[69, :, :], 'gray')

    plt.subplot(223)
    plt.imshow(img_0[75, :, :], 'gray')
    plt.subplot(224)
    plt.imshow(b_dilate[75, :, :], 'gray')


    # b = GetLabelMask(a[1], img_ori.shape)
    # plt.figure()
    # plt.subplot(221)
    # plt.imshow(b[279, :, :], 'gray')
    # plt.subplot(222)
    # plt.imshow(b[299, :, :], 'gray')
    # plt.subplot(223)
    # plt.imshow(img_ori[279, :, :], 'gray')
    # plt.subplot(224)
    # plt.imshow(img_ori[299, :, :], 'gray')


    # lbl_img = np.zeros(img_ori.shape)
    # for i in range(len(b.transpose())):
    #     lbl_img[b[2, i], b[0, i], b[1, i]] = 1
    # bb = lbl_img
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # #    fig = plt.figure()
    # ##    ax = plt.subplot(111, projection = '3d')
    # #    ax = fig.gca(projection="3d", title="plot title")
    # # bb = bb.reshape(3, -1)
    # # bb = bb.transpose()
    # ax.scatter(bb[0, :], bb[1, :], bb[2, :])
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    # plt.show()
    # lbl_img = resample(lbl_img, ori_spacing)
    # x, y, z = lbl_img.shape
    # bb = []
    # for iz in range(z):
    #     for ix in range(x):
    #         for iy in range(y):
    #             if lbl_img[iz, ix, iy] == 1:
    #                 bb = np.append(bb, [ix, iy, iz])
#     fig = plt.figure()
#
#     ax = fig.add_subplot(111, projection='3d')
# #    fig = plt.figure()
# ##    ax = plt.subplot(111, projection = '3d')
# #    ax = fig.gca(projection="3d", title="plot title")
#     b = lbl_img
#     # bb = bb.reshape(3, -1)
#     # bb = bb.transpose()
#     ax.scatter(bb[0,:], bb[1,:], bb[2,:])
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.set_zlabel('z')
#     plt.show()
