import torch
import numpy as np
from torch import einsum
from torch import Tensor
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg
from scipy.spatial.distance import directed_hausdorff

from typing import Any, Callable, Iterable, List, Set, Tuple, TypeVar, Union
torch.set_printoptions(precision=2)

# switch between representations
def probs2class(probs: Tensor) -> Tensor:
    b, _, w, h = probs.shape  # type: Tuple[int, int, int, int]
    assert simplex(probs)

    res = probs.argmax(dim=1)
    assert res.shape == (b, w, h)

    return res


def probs2one_hot(probs: Tensor) -> Tensor:
    _, C, _, _ = probs.shape
    assert simplex(probs)

    res = class2one_hot(probs2class(probs), C)
    assert res.shape == probs.shape
    assert one_hot(res)

    return res


def class2one_hot(seg: Tensor, C: int) -> Tensor:
    if len(seg.shape) == 2:  # Only w, h, used by the dataloader
        seg = seg.unsqueeze(dim=0)
    assert sset(seg, list(range(C)))

    b, w, h = seg.shape  # type: Tuple[int, int, int]

    res = torch.stack([seg == c for c in range(C)], dim=1).type(torch.int32)
    assert res.shape == (b, C, w, h)
    assert one_hot(res)

    return res


def one_hot2dist(seg: np.ndarray) -> np.ndarray:
    assert one_hot(torch.Tensor(seg), axis=0)
    C: int = len(seg)

    res = np.zeros_like(seg)
    for c in range(C):
        posmask = seg[c].astype(np.bool)

        if posmask.any():
            negmask = ~posmask
            print('negmask:', negmask)
            print('distance(negmask):', distance(negmask))
            res[c] = distance(negmask) * negmask - distance(posmask) * posmask
            print('res[c]', res[c])
    return res


def simplex(t: Tensor, axis=1) -> bool:
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)


def one_hot(t: Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])

    # Assert utils


def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())


def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)


class SurfaceLoss():
    def __init__(self):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = [1]  # 这里忽略背景类  https://github.com/LIVIAETS/surface-loss/issues/3

    # probs: bcwh, dist_maps: bcwh
    def __call__(self, probs: Tensor, dist_maps: Tensor, _: Tensor) -> Tensor:
        assert simplex(probs)
        assert not one_hot(dist_maps)

        pc = probs[:, self.idc, ...].type(torch.float32)
        dc = dist_maps[:, self.idc, ...].type(torch.float32)

        # print('pc', pc)
        # print('dc', dc)
        print('shape pc', pc.shape)
        print('shape dc', dc.shape)
        print('ORI DISTANCE MAP: ', dc)
        multipled = einsum("bcwh,bcwh->bcwh", pc, dc)

        loss = multipled.mean()

        return multipled#loss

def ComputeBinaryDistance(gt):
    #in distance, 0 means borders, 1 means areas out of borders
    gt_shape = gt.shape
    batch_num = gt_shape[0]
    channel_num = gt_shape[1]
    distance_map = np.zeros(gt_shape)
    for b in range(batch_num):
        for c in range(channel_num):
            distance_map[b][c] = distance(1 - gt[b][c]) - distance(gt[b][c])
    return distance_map



class BinaryBoundaryLoss(object):
    def __call__(self, pred, distance_map):
        assert distance_map.shape == pred.shape
        loss = pred * distance_map
        return loss.mean()


def compute_sdf(img_gt, out_shape):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the Signed Distance Map (SDM)
    sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    """

    img_gt = img_gt.astype(np.uint8)

    gt_sdf = np.zeros(out_shape)

    for b in range(out_shape[0]): # batch size
        for c in range(1, out_shape[1]):
            posmask = img_gt[b]
            print('posmark shape',posmask.shape)
            negmask = 1-posmask
            posdis = distance(posmask)
            negdis = distance(negmask)
            boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
            sdf = negdis - posdis
            sdf[boundary==1] = 0
            gt_sdf[b][c] = sdf

    return gt_sdf

def boundary_loss(outputs_soft, gt_sdf):
    """
    compute boundary loss for binary segmentation
    input: outputs_soft: softmax results,  shape=(b,2,x,y,z)
           gt_sdf: sdf of ground truth (can be original or normalized sdf); shape=(b,2,x,y,z)
    output: boundary_loss; sclar
    """
    pc = outputs_soft[:,[1],...]
    dc = gt_sdf[:,[1],...]
    multipled = einsum('bxyz, bxyz->bxyz', pc, dc)
    bd_loss = multipled.mean()

    print('MOD DISTANCE MAP: ', dc)

    # return bd_loss
    return multipled


if __name__ == "__main__":

    data = torch.tensor([[[0, 0, 0, 0, 0, 0, 0],
                          [0, 1, 1, 1, 0, 0, 0],
                          [0, 1, 1, 1, 0, 0, 0],
                          [0, 1, 1, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0]]])

    data2 = class2one_hot(data, 2)
    # print(data2)
    data2 = data2[0].numpy()
    data3 = one_hot2dist(data2)  # bcwh

    # print(data3)
    print("data3.shape:", data3.shape)

    logits_ori = torch.tensor([[[0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 1, 1, 1, 1, 0],
                            [0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 1]]])

    logits = class2one_hot(logits_ori, 2)

    Loss = SurfaceLoss()
    data3 = torch.tensor(data3).unsqueeze(0)

    res = Loss(logits, data3, None)
    print('loss:', res)
###
    bbl = BinaryBoundaryLoss()
    dc = ComputeBinaryDistance(data.unsqueeze(0))
    dc = torch.from_numpy(dc)
    res2 = bbl(logits_ori.unsqueeze(0), dc)
    print('loss2:', res2)
###
    gt_sdf_npy = compute_sdf(data2, data3.shape)
    gt_sdf = torch.from_numpy(gt_sdf_npy).float()
    loss_boundary = boundary_loss(logits, gt_sdf)
    print('loss3:', loss_boundary)