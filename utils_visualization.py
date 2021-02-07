from utils import GetDataEDAfromCSV, GetOrignalPixels, GetLabelCoor, GetLabelMask
from pathlib import Path
from matplotlib import pyplot as plt
from skimage.morphology.binary import binary_closing
from skimage.morphology import ball
from skimage.morphology import skeletonize_3d
import numpy as np
from skimage.measure import label as Label
from scipy.interpolate import interp1d as interp1d
import time
from PIL import Image

def largestConnectComponent(bw_img, img):
    t = time.time()
    labeled_img, num = Label(bw_img, neighbors=8, background=0, return_num=True)
    print('labeling connect area elapsed: ', round(time.time() - t, 3))
    t = time.time()
    assert num > 1, '神经管通常有两条'
    print('connect area # :', num)
    connnect_area = np.zeros(num + 1)
    for i in range(1, num + 1):  # in avoid of bg
        connnect_area[i] = np.sum(labeled_img == i)
    sort_idx = np.argsort(connnect_area)
    max_label = np.where(sort_idx == num)
    secondmax_label = np.where(sort_idx == num - 1)
    # lcc = (labeled_img == max_label)
    lcc = (labeled_img == secondmax_label)
    print('counting connect area elapsed: ', round(time.time() - t, 3))
    VisResult(lcc, img)
    # return lcc



class SaveImg(object):
    def __init__(self, save_dir=Path('save')):
        self.save_dir = save_dir

    def __call__(self, func):  # 接受函数
        def wrapper(*args, **kwargs):
            save_name = time.strftime('%Y_%m_%d_%H_%M_%S') + '.jpg'
            img = func(*args, **kwargs)
            img = Image.fromarray(img)
            save_path = Path(self.save_dir).joinpath(save_name)
            img.save(save_path)
        return wrapper  # 返回函数

@SaveImg(save_dir=Path('save'))
def VisResult(pred, label, img):
    t = time.time()
    label_centerline = skeletonize_3d(label)
    print('skeletonization elapsed: ', round(time.time() - t, 3))
    t = time.time()
    c = label_centerline.sum(axis=0)
    idx_tuple = np.nonzero(label_centerline)
    # skeleton_coor = list(zip(idx_tuple))
    x = np.array(idx_tuple[2])
    y = np.array(idx_tuple[1])

    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]
    z1 = np.polyfit(x, y, 7)
    print('fitting line elapsed: ', round(time.time() - t, 3))
    t = time.time()
    p1 = np.poly1d(z1)
    x_all = np.arange(label.shape[2])
    y_fit = p1(x_all)
    y_fit[y_fit < 0] = 0
    y_fit[y_fit > label.shape[1] - 1] = label.shape[1] - 1

    x_diff = np.diff(x_all)
    y_diff = np.diff(y_fit)
    d_length = [np.sqrt(i**2 + j**2) for i, j in zip(x_diff, y_diff)]
    d_length = np.hstack((0, d_length))
    cumlength = np.cumsum(d_length)

    space = 1
    l_sample = np.linspace(0, cumlength[-1], int(cumlength[-1] / space))
    fx = interp1d(cumlength, x_all, kind='cubic')
    xr = np.round(fx(l_sample)).astype(np.int16)
    xr[xr < 0] = 0
    fy = interp1d(cumlength, y_fit, kind='cubic')
    yr = np.round(fy(l_sample)).astype(np.int16)
    yr[yr < 0] = 0

    print('straighten line elapsed: ', round(time.time() - t, 3))
    t = time.time()

    img_show = img[:, yr, xr]
    img_show = HistEqualization(img_show)
    label_show = label[:, yr, xr].astype(np.uint8)
    pred_show = pred[:, yr, xr].astype(np.uint8)
    img_show[label_show == 1] = 0
    img_show[pred_show == 1] = 0
    img_show = np.tile(img_show[:, :, np.newaxis], 3)

    # label_show = (label[:, yr, xr] * 255).astype(np.uint8)
    img_show[:, :, 0][label_show == 1] = 255
    img_show[:, :, 1][pred_show == 1] = 255
    # img_show = np.concatenate((img_show, label_show), axis=-1)
    # plt.imshow(img_show, 'gray')
    # plt.plot(cumlength)
    return img_show

    # plt.plot(x_all, y_fit, 'r', label='polyfit')
    # plt.plot(x, y, '*', label='original values')
    # plt.show()

def HistEqualization(img, min_val=-1000, max_val=2200):
    img[img < min_val] = min_val
    img[img > max_val] = max_val
    img = ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    return img



# def SaveImg(save_dir=Path('save')):
#     '''
#     decorate
#     '''
#     pass
#     def wrapper(func):
#         def inner_wrapper(*args, **kwargs):
#             return func(*args, **kwargs)
#         return inner_wrapper
#     return wrapper

if __name__ == '__main__':



    csvinfo = GetDataEDAfromCSV(Path(r'data.csv'))
    data = csvinfo[3]
    t = time.time()
    img, ori_spacing = GetOrignalPixels(data)
    print('loading image elapsed: ', round(time.time() - t, 3))
    t = time.time()
    label = GetLabelMask(data, img.shape)
    print('loading label elapsed: ', round(time.time() - t, 3))
    t = time.time()
    stem = ball(5)
    label_closing = binary_closing(label, stem)
    print('binary operation elapsed: ', round(time.time() - t, 3))
    t = time.time()

    # labeled_img, num = Label(label_closing, neighbors=8, background=0, return_num=True)
    # label_closing = largestConnectComponent(label_closing, img)
    #
    label_closing = VisResult(label_closing, label_closing, img)


    # label_centerline = skeletonize_3d(label_closing)
    # c = label_centerline.sum(axis=0)
    # idx_tuple = np.nonzero(label_centerline)
    # # skeleton_coor = list(zip(idx_tuple))
    # y = np.array(idx_tuple[1])
    # x = np.array(idx_tuple[2])
    #
    # sort_idx = np.argsort(x)
    # x = x[sort_idx]
    # y = y[sort_idx]
    # z1 = np.polyfit(x, y, 7)
    # p1 = np.poly1d(z1)
    # x_all = np.arange(img.shape[2])
    # y_fit = p1(x_all)
    # y_fit[y_fit < 0] = 0
    # y_fit[y_fit > img.shape[1] - 1] = img.shape[1] - 1
    # plt.plot(x_all, y_fit, 'r', label='polyfit')
    # plt.plot(x, y, '*', label='original values')
    # plt.show()
    # plt.imshow(c, 'gray')
    # a = label.sum(axis=0)
    # b = label_closing.sum(axis=0)
    # plt.subplot(221)
    # plt.imshow(a, 'gray')
    # plt.subplot(222)
    # plt.imshow(b, 'gray')
    # plt.imshow(img[:, :, 222], 'gray')