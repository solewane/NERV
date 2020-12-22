from pathlib import Path
from utils import GetDataEDAfromCSV, GetOrignalPixels, GetLabelMask
import numpy as np
import scipy.ndimage
from scipy.ndimage.filters import gaussian_filter
from matplotlib import pyplot as plt
from scipy import ndimage
import time


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
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    real_new_spacing = ori_spacing / real_resize_factor
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest', order=interp_order)
    # when order = 0: nearest
    return image, real_new_spacing


def Dilate3D(image, ori_spacing, new_resolution):
    '''
    !!!EQUIVALENT TO  utils_transforms.RandomBinaryOperatorTransform!!!

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


def PadNDImamge(image, new_shape=None, mode="constant", kwargs=None, return_slicer=False, shape_must_be_divisible_by=None):
    """
    one padder to pad them all. Documentation? Well okay. A little bit
    :param image: nd image. can be anything
    :param new_shape: what shape do you want? new_shape does not have to have the same dimensionality as image. If
    len(new_shape) < len(image.shape) then the last axes of image will be padded. If new_shape < image.shape in any of
    the axes then we will not pad that axis, but also not crop! (interpret new_shape as new_min_shape)
    Example:
    image.shape = (10, 1, 512, 512); new_shape = (768, 768) -> result: (10, 1, 768, 768).
    image.shape = (10, 1, 512, 512); new_shape = (364, 768) -> result: (10, 1, 512, 768).
    :param mode: see np.pad for documentation
    :param return_slicer: if True then this function will also return what coords you will need to use when cropping back
    to original shape
    :param shape_must_be_divisible_by: for network prediction. After applying new_shape, make sure the new shape is
    divisibly by that number (can also be a list with an entry for each axis). Whatever is missing to match that will
    be padded (so the result may be larger than new_shape if shape_must_be_divisible_by is not None)
    :param kwargs: see np.pad for documentation
    """
    if kwargs is None:
        kwargs = {'constant_values': 0}

    if new_shape is not None:
        old_shape = np.array(image.shape[-len(new_shape):])
    else:
        assert shape_must_be_divisible_by is not None
        assert isinstance(shape_must_be_divisible_by, (list, tuple, np.ndarray))
        new_shape = image.shape[-len(shape_must_be_divisible_by):]
        old_shape = new_shape

    num_axes_nopad = len(image.shape) - len(new_shape)

    new_shape = [max(new_shape[i], old_shape[i]) for i in range(len(new_shape))]

    if not isinstance(new_shape, np.ndarray):
        new_shape = np.array(new_shape)

    if shape_must_be_divisible_by is not None:
        if not isinstance(shape_must_be_divisible_by, (list, tuple, np.ndarray)):
            shape_must_be_divisible_by = [shape_must_be_divisible_by] * len(new_shape)
        else:
            assert len(shape_must_be_divisible_by) == len(new_shape)

        for i in range(len(new_shape)):
            if new_shape[i] % shape_must_be_divisible_by[i] == 0:
                new_shape[i] -= shape_must_be_divisible_by[i]

        new_shape = np.array([new_shape[i] + shape_must_be_divisible_by[i] - new_shape[i] % shape_must_be_divisible_by[i] for i in range(len(new_shape))])

    difference = new_shape - old_shape
    pad_below = difference // 2
    pad_above = difference // 2 + difference % 2
    pad_list = [[0, 0]]*num_axes_nopad + list([list(i) for i in zip(pad_below, pad_above)])

    if not ((all([i == 0 for i in pad_below])) and (all([i == 0 for i in pad_above]))):
        res = np.pad(image, pad_list, mode, **kwargs)
    else:
        res = image

    if not return_slicer:
        return res
    else:
        pad_list = np.array(pad_list)
        pad_list[:, 1] = np.array(res.shape) - pad_list[:, 1]
        slicer = list(slice(*i) for i in pad_list)
        return res, slicer


def GetStepsForSlidingWindow(patch_size, image_size, step_size):
    assert [i >= j for i, j in zip(image_size, patch_size)], "image size must be as large or larger than patch_size"
    assert 0 < step_size <= 1, 'step_size must be larger than 0 and smaller or equal to 1'

    # our step width is patch_size*step_size at most, but can be narrower. For example if we have image size of
    # 110, patch size of 32 and step_size of 0.5, then we want to make 4 steps starting at coordinate 0, 27, 55, 78
    target_step_sizes_in_voxels = [i * step_size for i in patch_size]

    num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in zip(image_size, target_step_sizes_in_voxels, patch_size)]

    steps = []
    for dim in range(len(patch_size)):
        # the highest step value for this dimension is
        max_step_value = image_size[dim] - patch_size[dim]
        if num_steps[dim] > 1:
            actual_step_size = max_step_value / (num_steps[dim] - 1)
        else:
            actual_step_size = 99999999999  # does not matter because there is only one step at 0

        steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]

        steps.append(steps_here)

    return steps


def GetGaussianWeight(patch_size, sigma_scale=1./4) -> np.ndarray:
    tmp = np.zeros(patch_size)
    center_coords = [i // 2 for i in patch_size]
    sigmas = [i * sigma_scale for i in patch_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
    gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
    gaussian_importance_map = gaussian_importance_map.astype(np.float32)

    # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    gaussian_importance_map[gaussian_importance_map == 0] = np.min(
        gaussian_importance_map[gaussian_importance_map != 0])

    return gaussian_importance_map





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
    '''
    test difference of 2 dilate

    from utils_transforms import RandomBinaryOperatorTransform

    RandomBinaryOperatorTransform = RandomBinaryOperatorTransform(p_per_sample=1,
                                                                  strel_size=1)

    save_csv_path = Path(r'data.csv')
    a = GetDataEDAfromCSV(save_csv_path)
    img_ori, ori_spacing = GetOrignalPixels(a[1])
    b = GetLabelMask(a[1], img_ori.shape)
    time1 = time.time()
    # b_dilate = Dilate3D(b, ori_spacing, 1.)
    # b_dilate, real_new_spacing = Resample(b_dilate, ori_spacing, new_spacing=1., interp_order=0)
    time2 = time.time()
    img_0, real_new_spacing = Resample(b, ori_spacing, new_spacing=1., interp_order=0)
    b_dilate = RandomBinaryOperatorTransform(img_0)
    # img_0 = Dilate3DUsingCubicKernel(img_0, 1., 1.)
    img_0 = Dilate3D(img_0, 1., 1.)

    time3 = time.time()
    print('resample -> dilate : {:.2f}s, dilate -> resample: {:.2f}s'.format(time3-time2, time2-time1))
    print(img_0.sum())
    print(b_dilate.sum())
    plt.figure()

    plt.subplot(221)
    plt.imshow(img_0[69, :, :], 'gray')
    plt.subplot(222)
    plt.imshow(b_dilate[69, :, :], 'gray')

    plt.subplot(223)
    plt.imshow(img_0[75, :, :], 'gray')
    plt.subplot(224)
    plt.imshow(b_dilate[75, :, :], 'gray')
    '''
    #
    # save_csv_path = Path(r'data.csv')
    # img_name = '1mm.npy'
    # lbl_name = '1mm_mask_no_dilate.npy'
    #
    # a = GetDataEDAfromCSV(save_csv_path)
    # # img_list = [Path(i['Path']).joinpath(img_name) for i in a]
    # # lbl_list = [Path(i['Path']).joinpath(lbl_name) for i in a]
    # data_list = [[Path(i['Path']).joinpath(img_name), Path(i['Path']).joinpath(lbl_name)] for i in a]
    # data_list = [data_list[2]]
    # img_in = np.load(data_list[0][0])
    # img_out = PadNDImamge(img_in, (128, 128, 128))
    # lbl_in = np.load(data_list[0][1])
    # lbl_out = PadNDImamge(lbl_in, (128, 128, 128))
    # print(img_in.shape)
    # print(img_out.shape)
    # c = GetStepsForSlidingWindow((128, 128, 128), lbl_out.shape, 0.5)

    a= GetGaussianWeight((128, 128, 128))

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

    print('hello')