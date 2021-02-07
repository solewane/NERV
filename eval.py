
import torch
import numpy as np
from collections import OrderedDict
from scipy.ndimage import distance_transform_edt as distance

def Outline(image, mask, color):
    '''
    visualization
    Args:
        image: numpy size [h, w]
        mask: numpy size [h, w]
        color: outline color

    Return:
        image: numpy size [h, w ,c]
    '''
    image = (image + 1) / 2  # norm from [-1, 1] to [0, 1]
    image = np.expand_dims(image, 2)
    image = np.tile(image, (1, 1, 3)) / np.max(image)
    mask = np.round(mask)
    yy, xx = np.nonzero(mask)
    for y, x in zip(yy, xx):
        if 0.0 < np.mean(mask[max(0, y - 1):y + 2, max(0, x - 1):x + 2]) < 1.0:
            image[max(0, y):y + 1, max(0, x):x + 1] = color
    return image


def OutlineResults(image, pred, label, threshold=0.8):
    '''
    visualization, red : prediction , green : true label
    Args:
        image: numpy size [h, w]
        pred: numpy size [h, w]
        label: numpy size [h, w]
        threshold: true while > threshold

    Return:
        image: numpy size [h, w ,c]
    '''
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    pred = pred > threshold
    label = label > threshold
    # image[pred] = 0
    # image[label] = 0
    image = np.expand_dims(image, 2)
    image = np.tile(image, (1, 1, 3))
    yy, xx = np.nonzero(pred)
    for y, x in zip(yy, xx):
        if 0.0 < np.mean(pred[max(0, y - 1):y + 2, max(0, x - 1):x + 2]) < 1.0:
            image[max(0, y):y + 1, max(0, x):x + 1, 0] = 1
    yy, xx = np.nonzero(label)
    for y, x in zip(yy, xx):
        if 0.0 < np.mean(label[max(0, y - 1):y + 2, max(0, x - 1):x + 2]) < 1.0:
            image[max(0, y):y + 1, max(0, x):x + 1, 1] = 1

    return image


def CalDiceScore(label, pred, threshold=0.5):
    assert label.shape == pred.shape, print('the shape should match')
    epsilon = 1
    pred_copy = pred.copy()
    pred_copy[pred_copy >= threshold] = 1
    pred_copy[pred_copy < threshold] = 0
    map_TP = label * pred_copy
    FP = (pred - map_TP).sum()
    FN = (label - map_TP).sum()
    TP = map_TP.sum()
    precision = round(TP / (TP + FP), ndigits=3)
    recall = round(TP / (TP + FN), ndigits=3)
    numerator = 2 * TP
    denominator = label.sum() + pred_copy.sum() + epsilon
    dice_score = round(numerator / denominator, ndigits=3)
    return dice_score, int(TP), precision, recall


def TestThreshold(label_path, pred_path, threshold_list=[0.5, 0.6, 0.7, 0.8, 0.9], ndigits=3):
    assert len(label_path) == len(pred_path), 'the length should match'
    dice_all = []
    for idx in range(len(label_path)):
        label = np.load(label_path[idx])
        pred = np.load(pred_path[idx])
        assert label.shape == pred.shape, 'the shape should match'
        dice_sample = []
        for threshold in threshold_list:
            tmp = CalDiceScore(label, pred, threshold=threshold)
            dice_sample.append(round(tmp, ndigits=ndigits))
        dice_all.append(dice_sample)
    return dice_all


def CalBoundaryScore(label, pred, threshold=0.5):
    '''
    customize metric
    distance map
    '''
    assert label.shape == pred.shape
    pred_copy = pred.copy()
    pred_copy[pred_copy >= threshold] = 1
    pred_copy[pred_copy < threshold] = 0
    dismap = distance(1 - label)
    boundary_FP = round((dismap * pred_copy).sum())
    dismap = distance(label)
    boundary_FN = round((dismap * (1 - pred_copy)).sum())
    return boundary_FP, boundary_FN



if __name__ == '__main__':
    print('sss')
    label = np.array([[[0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 1, 1, 0, 0, 0],
                      [0, 1, 1, 1, 0, 0, 0],
                      [0, 1, 1, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0]]])


    pred = np.array([[[0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 1, 1, 1, 1, 0],
                            [0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 1]]])

    a, b, c, d = CalDiceScore(label, pred)
    e, f = CalBoundaryScore(label, pred)