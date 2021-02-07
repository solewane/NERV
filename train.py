import torch
import torch.nn as nn
import models
from utils_dataset import CoarseResolutionDataset
from pathlib import Path
from torch.utils.data import DataLoader
from utils import GetDataEDAfromCSV
from torch.utils.tensorboard import SummaryWriter
import nnunet
import numpy as np
import random
from sklearn.model_selection import KFold
from tqdm import tqdm
import time
from collections import OrderedDict
from utils_transforms import Normalize
from utils_images import PadNDImamge, GetStepsForSlidingWindow, GetGaussianWeight
from eval import TestThreshold, CalDiceScore, CalBoundaryScore
from loss_function import ComputeBinaryDistance, BinaryBoundaryLoss
from utils_visualization import VisResult


def ThrowDice(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子


def CheckpointSave(model, optimizer, epoch, path):
    state = {'model': model.module.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
    torch.save(state, path)


def Save_Checkpoint(network, epoch, fname):
    start_time = time.time()
    state_dict = network.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()
    print("saving checkpoint...")
    save_this = {
        'epoch': epoch + 1,
        'state_dict': state_dict}
    torch.save(save_this, fname)
    print("done, saving took %.2f seconds" % (time.time() - start_time))


def Load_Checkpoint(network, fname):
    """
    """
    checkpoint = torch.load(fname, map_location=torch.device('cpu'))
    new_state_dict = OrderedDict()
    curr_state_dict_keys = list(network.state_dict().keys())
    # if state dict comes form nn.DataParallel but we use non-parallel model here then the state dict keys do not
    # match. Use heuristic to make it match
    for k, value in checkpoint['state_dict'].items():
        key = k
        if key not in curr_state_dict_keys and key.startswith('module.'):
            key = key[7:]
        new_state_dict[key] = value

    network.load_state_dict(new_state_dict)
    epoch = checkpoint['epoch']


def KFoldSplit(data_list, seed=24):
    kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
    splits = []
    for i, (train_idx, test_idx) in enumerate(kfold.split(data_list)):
        splits.append({'train': np.array(data_list)[train_idx], 'val': np.array(data_list)[test_idx]})
    return splits


# if torch.cuda.device_count() > 1:  # MUTLI-GPUS
#     # torch.distributed.init_process_group(backend="nccl")
#     torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:23456', rank=0, world_size=1)
#     print("Let's use", torch.cuda.device_count(), "GPUs!")
#     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
#     net = net.to('cuda')
#     net = nn.parallel.DistributedDataParallel(net, find_unused_parameters=True)
#     # net = nn.DataParallel(net)

def Poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
    return initial_lr * (1 - epoch / max_epochs) ** exponent


@torch.no_grad()
def Pred3DTiled(net, data_list,
                step_size=0.5, patch_size=(128, 128, 128),
                min_ctnum=-1000, max_ctnum=2200,
                use_gaussian_weight=True, sigma_scale=1. /8,
                is_save=True, save_name='label_pred.npy'):
    dice_result_list = []
    precison_list = []
    recall_list = []
    label_voxel_list = []
    bd_fp_list = []
    bd_fn_list = []
    bd_all_list = []
    ThrowDice(24)
    device = torch.device('cuda')
    net.to(device)
    net.eval()
    torch.backends.cudnn.benchmark = True  # pre GPU optimization
    criterion = models.BinaryDiceLoss()
    if use_gaussian_weight:
        weight = GetGaussianWeight(patch_size, sigma_scale=sigma_scale)
    else:
        weight = np.ones(patch_size)

    for idx in range(len(data_list)):
        img_fullsize = np.load(data_list[idx][0]).astype(np.float32)
        lbl_fullsize = np.load(data_list[idx][1])
        norm_method = Normalize()
        img_fullsize = norm_method(img_fullsize, min_ctnum, max_ctnum)
        img_fullsize = img_fullsize * 2 - 1
        img_fullsize = PadNDImamge(img_fullsize, patch_size)
        lbl_fullsize, slicer = PadNDImamge(lbl_fullsize, patch_size, return_slicer=True)
        img_fullsize = torch.from_numpy(img_fullsize)
        lbl_fullsize = torch.from_numpy(lbl_fullsize)
        weight_fullsize = np.zeros(lbl_fullsize.shape)
        pred_fullsize = np.zeros(lbl_fullsize.shape)
        steps = GetStepsForSlidingWindow(patch_size, lbl_fullsize.shape, step_size)
        for z in steps[0]:
            lb_z = z
            ub_z = z + patch_size[0]
            for x in steps[1]:
                lb_x = x
                ub_x = x + patch_size[1]
                for y in steps[2]:
                    lb_y = y
                    ub_y = y + patch_size[2]

                    img_patch = img_fullsize[lb_z:ub_z, lb_x:ub_x, lb_y:ub_y].unsqueeze(0).unsqueeze(0).to(device)
                    predicted_patch = net(img_patch)[0].to('cpu').squeeze(0).squeeze(0).numpy()

                    weight_fullsize[lb_z:ub_z, lb_x:ub_x, lb_y:ub_y] += weight
                    pred_fullsize[lb_z:ub_z, lb_x:ub_x, lb_y:ub_y] += predicted_patch * weight
        pred_fullsize = pred_fullsize / weight_fullsize

        dice_score, _, precision, recall = CalDiceScore(lbl_fullsize.numpy(), pred_fullsize)
        bd_fp, bd_fn = CalBoundaryScore(lbl_fullsize.numpy(), pred_fullsize)
        # dice_result_fullsize = round(1 - criterion(torch.from_numpy(pred_fullsize).unsqueeze(0), lbl_fullsize.unsqueeze(0)).item(), 3)
        # print(dice_result_fullsize)
        # print(CalDiceScore(pred_fullsize, lbl_fullsize.numpy()))
        dice_result_list.append(dice_score)
        precison_list.append(precision)
        recall_list.append(recall)
        label_voxel_list.append(lbl_fullsize.sum())
        bd_fp_list.append(bd_fp)
        bd_fn_list.append(bd_fn)
        bd_all_list.append(bd_fp + bd_fn)
        '''
        visualization
        pred & label shape is the shape after padding
        '''
        img_fullsize = np.load(data_list[idx][0]).astype(np.float32)
        # print('img shape:', img_fullsize.shape)
        # print('pred shape:', pred_fullsize.shape)
        # print('lbl shape:', lbl_fullsize.numpy().shape)
        pred_fullsize[pred_fullsize > 0.5] = 1
        pred_fullsize[pred_fullsize <= 0.5] = 0
        VisResult(pred_fullsize[slicer], lbl_fullsize.numpy()[slicer], img_fullsize)
        if is_save:
            np.save(data_list[idx][1].parent.joinpath(save_name), pred_fullsize[slicer])
            # print(data_list[idx][1].parent.joinpath(save_name))
    return dice_result_list, label_voxel_list, precison_list, recall_list, bd_fp_list, bd_fn_list, bd_all_list


def main_train():
    checkpoint_flag = False
    checkpoint_model = None
    ThrowDice(24)
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True  # pre GPU optimization
    save_csv_path = Path(r'data.csv')
    img_name = '1mm.npy'
    # lbl_name = '1mm_mask_no_dilate.npy'
    lbl_name = '1mm_mask.npy'
    # lbl_name = '1mm_mask_closing_kernel3.npy'
    epoch_total = 3000
    alpha = 1 #boundary loss + dice loss
    init_lr = 0.01
    save_freq = 500
    writer = SummaryWriter('test_log')
    csv_info = GetDataEDAfromCSV(save_csv_path)
    data_list_all = [[Path(i['Path']).joinpath(img_name), Path(i['Path']).joinpath(lbl_name)] for i in csv_info]
    K_Folder = KFoldSplit(data_list_all)
    data_list = K_Folder[2]['train']  # k folder
    data_list_val = K_Folder[2]['val']
    n_train = len(data_list)
    n_val = len(data_list_val)
    train_dataset = CoarseResolutionDataset(pair_list=data_list,
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
                                            is_morphology=False,
                                            p_morphology=0.3,
                                            )
    val_dataset = CoarseResolutionDataset(pair_list=data_list_val,
                                          is_train=False,
                                          crop_size=(128, 128, 128),
                                          min_ctnum=-1000,
                                          max_ctnum=2200,
                                          is_croppadding=True
                                          )
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=2,
                                  shuffle=True,
                                  num_workers=1)
    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=1)
    # net = models.VanillaVNet().to(device)
    net = nnunet.Generic_UNet(input_channels=1, base_num_features=nnunet.Generic_UNet.BASE_NUM_FEATURES_3D,
                              num_classes=1, num_pool=5, num_conv_per_stage=2,
                              feat_map_mul_on_downscale=2, conv_op=nn.Conv3d,
                              norm_op=nn.InstanceNorm3d, norm_op_kwargs=None,
                              dropout_op=nn.Dropout3d,
                              dropout_op_kwargs=None,
                              nonlin=nn.LeakyReLU, nonlin_kwargs=None, deep_supervision=True,
                              dropout_in_localization=False,
                              final_nonlin=nn.Sigmoid(), weightInitializer=nnunet.InitWeights_He(1e-2),
                              pool_op_kernel_sizes=None,
                              conv_kernel_sizes=None,
                              upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
                              max_num_features=None, basic_block=nnunet.ConvDropoutNormNonlin,
                              seg_output_use_bias=False)
    net = net.to(device)
    criterion_DICE = models.BinaryDiceLoss()
    criterion_CE = nn.BCELoss()
    criterion_BOUNDARY = BinaryBoundaryLoss()
    optim = torch.optim.Adam(net.parameters(),
                             lr=init_lr,  # 3e-4,
                             betas=(0.9, 0.999),
                             weight_decay=0)
    epoch_start = 0
    if checkpoint_flag:
        checkpoint = torch.load(checkpoint_model)
        # net.load_state_dict(checkpoint['model'])
        net.load_state_dict({'module.' + k: v for k, v in checkpoint['model'].items()})
        # net.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['model'].items()})
        optim.load_state_dict(checkpoint['optimizer'])
        epoch_start = checkpoint['epoch']
        loss = checkpoint['loss']

    start_time = time.time()
    for epoch in range(epoch_start, epoch_total):
        net.train()
        for param_group in optim.param_groups:
            param_group['lr'] = Poly_lr(epoch, epoch_total, init_lr, exponent=0.9)
        loss_ce_item = 0
        loss_dice_item = 0
        loss_CE_last_item = 0
        loss_CE_loss_supervision1_item = 0
        loss_CE_loss_supervision2_item = 0
        loss_CE_loss_supervision3_item = 0
        loss_DICE_last_item = 0
        loss_DICE_loss_supervision1_item = 0
        loss_DICE_loss_supervision2_item = 0
        loss_DICE_loss_supervision3_item = 0
        print(epoch)
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epoch_total}', unit='img') as pbar:
            for iter_, (x, y, y2, y4, y8) in enumerate(train_dataloader):
                n_batch = x.shape[0]
                pbar.update(n_batch)
                x = x.to(device).unsqueeze(1).to(device)
                y = y.to(device).unsqueeze(1).float().to(device)
                y2 = y2.to(device).unsqueeze(1).float().to(device)
                y4 = y4.to(device).unsqueeze(1).float().to(device)
                y8 = y8.to(device).unsqueeze(1).float().to(device)
                pred, supervision1, supervision2, supervision3, _ = net(x)
                loss_CE_last = criterion_CE(pred, y)
                loss_CE_supervision1 = criterion_CE(supervision1, y2)
                loss_CE_supervision2 = criterion_CE(supervision2, y4)
                loss_CE_supervision3 = criterion_CE(supervision3, y8)
                loss_DICE_last = criterion_DICE(pred, y)
                loss_DICE_supervision1 = criterion_DICE(supervision1, y2)
                loss_DICE_supervision2 = criterion_DICE(supervision2, y4)
                loss_DICE_supervision3 = criterion_DICE(supervision3, y8)

                loss_CE = loss_CE_last + 0.5 * loss_CE_supervision1 + 0.25 * loss_CE_supervision2 + 0.125 * loss_CE_supervision3
                '''
                test boundary loss start
                '''
                distance_map = ComputeBinaryDistance(y.to('cpu').numpy())
                distance_map = torch.from_numpy(distance_map).to(device)
                loss_boundary = criterion_BOUNDARY(pred, distance_map)
                loss_DICE = alpha * (loss_DICE_last + 0.5 * loss_DICE_supervision1 + 0.25 * loss_DICE_supervision2 + 0.125 * loss_DICE_supervision3) \
                     + (1 - alpha) * loss_boundary
                '''
                test boundary loss end
                '''
                optim.zero_grad()
                loss_CE.backward(retain_graph=True)
                loss_DICE.backward()
                optim.step()
                # print(iter_)
                loss_ce_item += loss_CE.detach().item() * n_batch / n_train
                loss_dice_item += loss_DICE.detach().item() * n_batch / n_train
                # print('LOSS CE {}'.format(loss_ce_item))
                # print('LOSS DICE {}'.format(loss_dice_item))
                loss_CE_last_item += loss_CE_last.detach().item() * n_batch / n_train
                loss_CE_loss_supervision1_item += loss_CE_supervision1.detach().item() * n_batch / n_train
                loss_CE_loss_supervision2_item += loss_CE_supervision2.detach().item() * n_batch / n_train
                loss_CE_loss_supervision3_item += loss_CE_supervision3.detach().item() * n_batch / n_train
                loss_DICE_last_item += loss_DICE_last.detach().item() * n_batch / n_train
                loss_DICE_loss_supervision1_item += loss_DICE_supervision1.detach().item() * n_batch / n_train
                loss_DICE_loss_supervision2_item += loss_DICE_supervision2.detach().item() * n_batch / n_train
                loss_DICE_loss_supervision3_item += loss_DICE_supervision3.detach().item() * n_batch / n_train
        loss_ce_item_val = 0
        loss_dice_item_val = 0
        loss_CE_last_item_val = 0
        loss_CE_loss_supervision1_item_val = 0
        loss_CE_loss_supervision2_item_val = 0
        loss_CE_loss_supervision3_item_val = 0
        loss_DICE_last_item_val = 0
        loss_DICE_loss_supervision1_item_val = 0
        loss_DICE_loss_supervision2_item_val = 0
        loss_DICE_loss_supervision3_item_val = 0

        alpha -= 0.001
        if alpha < 0.001:
            alpha = 0.001
        net.eval()
        with torch.no_grad():
            with tqdm(total=n_val, desc=f'Epoch {epoch + 1}/{epoch_total}', unit='img') as pbar_val:
                for iter_, (x, y, y2, y4, y8) in enumerate(val_dataloader):
                    n_batch = x.shape[0]
                    pbar_val.update(n_batch)
                    x = x.to(device).unsqueeze(1).to(device)
                    y = y.to(device).unsqueeze(1).float().to(device)
                    y2 = y2.to(device).unsqueeze(1).float().to(device)
                    y4 = y4.to(device).unsqueeze(1).float().to(device)
                    y8 = y8.to(device).unsqueeze(1).float().to(device)
                    pred, supervision1, supervision2, supervision3, _ = net(x)
                    loss_CE_last = criterion_CE(pred, y)
                    loss_CE_supervision1 = criterion_CE(supervision1, y2)
                    loss_CE_supervision2 = criterion_CE(supervision2, y4)
                    loss_CE_supervision3 = criterion_CE(supervision3, y8)
                    loss_DICE_last = criterion_DICE(pred, y)
                    loss_DICE_supervision1 = criterion_DICE(supervision1, y2)
                    loss_DICE_supervision2 = criterion_DICE(supervision2, y4)
                    loss_DICE_supervision3 = criterion_DICE(supervision3, y8)
                    loss_CE = loss_CE_last + 0.5 * loss_CE_supervision1 + 0.25 * loss_CE_supervision2 + 0.125 * loss_CE_supervision3
                    loss_DICE = loss_DICE_last + 0.5 * loss_DICE_supervision1 + 0.25 * loss_DICE_supervision2 + 0.125 * loss_DICE_supervision3
                    # print(iter_)
                    loss_ce_item_val += loss_CE.detach().item() * n_batch / n_val
                    loss_dice_item_val += loss_DICE.detach().item() * n_batch / n_val
                    # print('LOSS CE {}'.format(loss_ce_item))
                    # print('LOSS DICE {}'.format(loss_dice_item))
                    loss_CE_last_item_val += loss_CE_last.detach().item() * n_batch / n_val
                    loss_CE_loss_supervision1_item_val += loss_CE_supervision1.detach().item() * n_batch / n_val
                    loss_CE_loss_supervision2_item_val += loss_CE_supervision2.detach().item() * n_batch / n_val
                    loss_CE_loss_supervision3_item_val += loss_CE_supervision3.detach().item() * n_batch / n_val
                    loss_DICE_last_item_val += loss_DICE_last.detach().item() * n_batch / n_val
                    loss_DICE_loss_supervision1_item_val += loss_DICE_supervision1.detach().item() * n_batch / n_val
                    loss_DICE_loss_supervision2_item_val += loss_DICE_supervision2.detach().item() * n_batch / n_val
                    loss_DICE_loss_supervision3_item_val += loss_DICE_supervision3.detach().item() * n_batch / n_val
        writer.add_scalars('loss_all',
                           {'training_loss_all': loss_ce_item + loss_dice_item,
                            'valid_loss_all': loss_ce_item_val + loss_dice_item_val, },
                           epoch)
        writer.add_scalars('loss_ce',
                           {'training_loss_last': loss_CE_last_item,
                            'training_loss_s1': loss_CE_loss_supervision1_item,
                            'training_loss_s2': loss_CE_loss_supervision2_item,
                            'training_loss_s3': loss_CE_loss_supervision3_item,
                            'valid_loss_last': loss_CE_last_item_val,
                            'valid_loss_s1': loss_CE_loss_supervision1_item_val,
                            'valid_loss_s2': loss_CE_loss_supervision2_item_val,
                            'valid_loss_s3': loss_CE_loss_supervision3_item_val},
                           epoch)
        writer.add_scalars('loss_dice',
                           {'training_loss_dice_last': loss_DICE_last_item,
                            'training_loss_dice_s1': loss_DICE_loss_supervision1_item,
                            'training_loss_dice_s2': loss_DICE_loss_supervision2_item,
                            'training_loss_dice_s3': loss_DICE_loss_supervision3_item,
                            'valid_loss_dice_last': loss_DICE_last_item_val,
                            'valid_loss_dice_s1': loss_DICE_loss_supervision1_item_val,
                            'valid_loss_dice_s2': loss_DICE_loss_supervision2_item_val,
                            'valid_loss_dice_s3': loss_DICE_loss_supervision3_item_val},
                           epoch)
        print('epoch:{},'
              'training_CE_loss:{:.4f},training_DiCE_loss:{:.4f},'
              'valid_CE_loss:{:.4f},valid_DiCE_loss:{:.4f},'
              'cost time:{:.2f}s'.format(epoch,
                                         loss_CE_last_item, loss_DICE_last_item,
                                         loss_CE_last_item_val, loss_DICE_last_item_val,
                                         time.time() - start_time))
        torch.cuda.empty_cache()
        if epoch % save_freq == 0:
            # path = 'save/epoch{}_CE{:.4f}_Dice{:.4f}.pth'.format(epoch, loss_CE_last_item, loss_DICE_last_item)
            # CheckpointSave(net, optim, epoch, path)
            path = 'save/epoch{}_CE{:.4f}_Dice{:.4f}.model'.format(epoch, loss_CE_last_item, loss_DICE_last_item)
            Save_Checkpoint(net, epoch, path)
    print('hello')


def main_val():
    dice_train = []
    dice_valid = []
    fname = r'/home/zhangyunxu/NERV/save/epoch1000_CE0.0099_Dice0.3138.model'
    net = nnunet.Generic_UNet(input_channels=1, base_num_features=nnunet.Generic_UNet.BASE_NUM_FEATURES_3D,
                              num_classes=1, num_pool=5, num_conv_per_stage=2,
                              feat_map_mul_on_downscale=2, conv_op=nn.Conv3d,
                              norm_op=nn.InstanceNorm3d, norm_op_kwargs=None,
                              dropout_op=nn.Dropout3d, dropout_op_kwargs=None,
                              nonlin=nn.LeakyReLU, nonlin_kwargs=None, deep_supervision=True,
                              dropout_in_localization=False,
                              final_nonlin=nn.Sigmoid(), weightInitializer=nnunet.InitWeights_He(1e-2),
                              pool_op_kernel_sizes=None,
                              conv_kernel_sizes=None,
                              upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
                              max_num_features=None, basic_block=nnunet.ConvDropoutNormNonlin,
                              seg_output_use_bias=False)
    Load_Checkpoint(net, fname)
    ThrowDice(24)
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True  # pre GPU optimization
    save_csv_path = Path(r'data.csv')
    img_name = '1mm.npy'
    lbl_name = '1mm_mask_no_dilate.npy'
    writer = SummaryWriter('test_log')
    csv_info = GetDataEDAfromCSV(save_csv_path)
    data_list_all = [[Path(i['Path']).joinpath(img_name), Path(i['Path']).joinpath(lbl_name)] for i in csv_info]
    K_Folder = KFoldSplit(data_list_all)
    data_list = K_Folder[0]['train']  # k folder
    data_list_val = K_Folder[0]['val']
    n_train = len(data_list)
    n_val = len(data_list_val)
    train_dataset = CoarseResolutionDataset(pair_list=data_list,
                                            is_train=False,
                                            crop_size=(128, 128, 128),
                                            min_ctnum=-1000,
                                            max_ctnum=2200,
                                            is_croppadding=True,
                                            )
    val_dataset = CoarseResolutionDataset(pair_list=data_list_val,
                                          is_train=False,
                                          crop_size=(128, 128, 128),
                                          min_ctnum=-1000,
                                          max_ctnum=2200,
                                          is_croppadding=True
                                          )
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=1)
    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=1)
    net = net.to(device)
    criterion_DICE = models.BinaryDiceLoss()
    criterion_CE = nn.BCELoss()
    start_time = time.time()
    with torch.no_grad():
        net.eval()
        test = 0
        loss_ce_item = 0
        loss_dice_item = 0
        loss_CE_last_item = 0
        loss_CE_loss_supervision1_item = 0
        loss_CE_loss_supervision2_item = 0
        loss_CE_loss_supervision3_item = 0
        loss_DICE_last_item = 0
        loss_DICE_loss_supervision1_item = 0
        loss_DICE_loss_supervision2_item = 0
        loss_DICE_loss_supervision3_item = 0
        with tqdm(total=n_train, desc=f'valid_train', unit='img') as pbar:
            for iter_, (x, y, y2, y4, y8) in enumerate(train_dataloader):
                n_batch = x.shape[0]
                pbar.update(n_batch)
                x = x.to(device).unsqueeze(1).to(device)
                y = y.to(device).unsqueeze(1).float().to(device)
                y2 = y2.to(device).unsqueeze(1).float().to(device)
                y4 = y4.to(device).unsqueeze(1).float().to(device)
                y8 = y8.to(device).unsqueeze(1).float().to(device)
                pred, supervision1, supervision2, supervision3, _ = net(x)
                loss_CE_last = criterion_CE(pred, y)
                loss_CE_supervision1 = criterion_CE(supervision1, y2)
                loss_CE_supervision2 = criterion_CE(supervision2, y4)
                loss_CE_supervision3 = criterion_CE(supervision3, y8)
                loss_DICE_last = criterion_DICE(pred, y)
                loss_DICE_supervision1 = criterion_DICE(supervision1, y2)
                loss_DICE_supervision2 = criterion_DICE(supervision2, y4)
                loss_DICE_supervision3 = criterion_DICE(supervision3, y8)

                loss_CE = loss_CE_last + 0.5 * loss_CE_supervision1 + 0.25 * loss_CE_supervision2 + 0.125 * loss_CE_supervision3
                loss_DICE = loss_DICE_last + 0.5 * loss_DICE_supervision1 + 0.25 * loss_DICE_supervision2 + 0.125 * loss_DICE_supervision3
                # print(iter_)
                loss_ce_item += loss_CE.detach().item() * n_batch / n_train
                loss_dice_item += loss_DICE.detach().item() * n_batch / n_train
                # print('LOSS CE {}'.format(loss_ce_item))
                # print('LOSS DICE {}'.format(loss_dice_item))
                loss_CE_last_item += loss_CE_last.detach().item() * n_batch / n_train
                loss_CE_loss_supervision1_item += loss_CE_supervision1.detach().item() * n_batch / n_train
                loss_CE_loss_supervision2_item += loss_CE_supervision2.detach().item() * n_batch / n_train
                loss_CE_loss_supervision3_item += loss_CE_supervision3.detach().item() * n_batch / n_train
                loss_DICE_last_item += loss_DICE_last.detach().item() * n_batch / n_train
                loss_DICE_loss_supervision1_item += loss_DICE_supervision1.detach().item() * n_batch / n_train
                loss_DICE_loss_supervision2_item += loss_DICE_supervision2.detach().item() * n_batch / n_train
                loss_DICE_loss_supervision3_item += loss_DICE_supervision3.detach().item() * n_batch / n_train
                dice_train.append(loss_DICE_last.detach().item())
        loss_ce_item_val = 0
        loss_dice_item_val = 0
        loss_CE_last_item_val = 0
        loss_CE_loss_supervision1_item_val = 0
        loss_CE_loss_supervision2_item_val = 0
        loss_CE_loss_supervision3_item_val = 0
        loss_DICE_last_item_val = 0
        loss_DICE_loss_supervision1_item_val = 0
        loss_DICE_loss_supervision2_item_val = 0
        loss_DICE_loss_supervision3_item_val = 0
        with tqdm(total=n_val, desc=f'valid_valid', unit='img') as pbar_val:
            for iter_, (x, y, y2, y4, y8) in enumerate(val_dataloader):
                n_batch = x.shape[0]
                pbar_val.update(n_batch)
                x = x.to(device).unsqueeze(1).to(device)
                y = y.to(device).unsqueeze(1).float().to(device)
                y2 = y2.to(device).unsqueeze(1).float().to(device)
                y4 = y4.to(device).unsqueeze(1).float().to(device)
                y8 = y8.to(device).unsqueeze(1).float().to(device)
                pred, supervision1, supervision2, supervision3, _ = net(x)
                loss_CE_last = criterion_CE(pred, y)
                loss_CE_supervision1 = criterion_CE(supervision1, y2)
                loss_CE_supervision2 = criterion_CE(supervision2, y4)
                loss_CE_supervision3 = criterion_CE(supervision3, y8)
                loss_DICE_last = criterion_DICE(pred, y)
                loss_DICE_supervision1 = criterion_DICE(supervision1, y2)
                loss_DICE_supervision2 = criterion_DICE(supervision2, y4)
                loss_DICE_supervision3 = criterion_DICE(supervision3, y8)
                loss_CE = loss_CE_last + 0.5 * loss_CE_supervision1 + 0.25 * loss_CE_supervision2 + 0.125 * loss_CE_supervision3
                loss_DICE = loss_DICE_last + 0.5 * loss_DICE_supervision1 + 0.25 * loss_DICE_supervision2 + 0.125 * loss_DICE_supervision3
                # print(iter_)
                loss_ce_item_val += loss_CE.detach().item() * n_batch / n_val
                loss_dice_item_val += loss_DICE.detach().item() * n_batch / n_val
                # print('LOSS CE {}'.format(loss_ce_item))
                # print('LOSS DICE {}'.format(loss_dice_item))
                loss_CE_last_item_val += loss_CE_last.detach().item() * n_batch / n_val
                loss_CE_loss_supervision1_item_val += loss_CE_supervision1.detach().item() * n_batch / n_val
                loss_CE_loss_supervision2_item_val += loss_CE_supervision2.detach().item() * n_batch / n_val
                loss_CE_loss_supervision3_item_val += loss_CE_supervision3.detach().item() * n_batch / n_val
                loss_DICE_last_item_val += loss_DICE_last.detach().item() * n_batch / n_val
                loss_DICE_loss_supervision1_item_val += loss_DICE_supervision1.detach().item() * n_batch / n_val
                loss_DICE_loss_supervision2_item_val += loss_DICE_supervision2.detach().item() * n_batch / n_val
                loss_DICE_loss_supervision3_item_val += loss_DICE_supervision3.detach().item() * n_batch / n_val
                dice_valid.append(loss_DICE_last.detach().item())
    print('epoch:{},'
          'training_CE_loss:{:.4f},training_DiCE_loss:{:.4f},'
          'valid_CE_loss:{:.4f},valid_DiCE_loss:{:.4f},'
          'cost time:{:.2f}s'.format('valid',
                                     loss_CE_last_item, loss_DICE_last_item,
                                     loss_CE_last_item_val, loss_DICE_last_item_val,
                                     time.time() - start_time))
    print(test)
    torch.cuda.empty_cache()
    return dice_train, dice_valid


if __name__ == '__main__':
    # a, b = main_val()
    '''
    train
    '''
    # main_train()
    '''
    valid_tiled
    '''

    fname = r'/home/zhangyunxu/NERV/save/epoch1000_CE0.0007_Dice0.1330.model'
    net = nnunet.Generic_UNet(input_channels=1, base_num_features=nnunet.Generic_UNet.BASE_NUM_FEATURES_3D,
                              num_classes=1, num_pool=5, num_conv_per_stage=2,
                              feat_map_mul_on_downscale=2, conv_op=nn.Conv3d,
                              norm_op=nn.InstanceNorm3d, norm_op_kwargs=None,
                              dropout_op=nn.Dropout3d,
                              dropout_op_kwargs=None,
                              nonlin=nn.LeakyReLU, nonlin_kwargs=None, deep_supervision=True,
                              dropout_in_localization=False,
                              final_nonlin=nn.Sigmoid(), weightInitializer=nnunet.InitWeights_He(1e-2),
                              pool_op_kernel_sizes=None,
                              conv_kernel_sizes=None,
                              upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
                              max_num_features=None, basic_block=nnunet.ConvDropoutNormNonlin,
                              seg_output_use_bias=False)
    Load_Checkpoint(net, fname)
    # net

    save_csv_path = Path(r'data.csv')
    img_name = '1mm.npy'
    # lbl_name = '1mm_mask_closing_kernel3.npy'
    lbl_name = '1mm_mask.npy'
    # lbl_name = '1mm_mask_no_dilate.npy'
    csv_info = GetDataEDAfromCSV(save_csv_path)
    data_list_all = [[Path(i['Path']).joinpath(img_name), Path(i['Path']).joinpath(lbl_name)] for i in csv_info]
    K_Folder = KFoldSplit(data_list_all)
    data_list = K_Folder[2]['train']  # k folder
    data_list_val = K_Folder[2]['val']
    # data list

    dice_result_list, label_voxel_list, precison_list, recall_list, bd_fp, bd_fn, bd_all_list = Pred3DTiled(net, data_list_val,
                         step_size=0.5, patch_size=(128, 128, 128), min_ctnum=-1000, max_ctnum=2200,
                         use_gaussian_weight=True, sigma_scale=1./8)
    print('dice score:')
    print(dice_result_list)
    print(np.array(dice_result_list).mean())
    print('label voxel:')
    print([i.item() for i in label_voxel_list])
    print(np.array(label_voxel_list).mean())
    print('precision:')
    print(precison_list)
    print(np.array(precison_list).mean())
    print('recall:')
    print(recall_list)
    print(np.array(recall_list).mean())
    print('bd_all:')
    print(bd_all_list)
    print(np.array(bd_all_list).mean())
    print('fp:')
    print(bd_fp)
    print(np.array(bd_fp).mean())
    print('fn:')
    print(bd_fn)
    print(np.array(bd_fn).mean())


    #
    # print('mean value')
    # print(np.array(result).mean())
    # # threshold
    # pred_name = 'label_pred_1mm_mask.npy'
    # label_list = [i[1] for i in data_list_val]
    # pred_list = [i.parent.joinpath(pred_name) for i in label_list]
    # pred_result = TestThreshold(label_list, pred_list, threshold_list=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    #
    # print(np.round(np.array(pred_result).mean(0), 3))

    '''
    test threshold of pred 
    '''
    # save_csv_path = Path(r'data.csv')
    # img_name = '1mm.npy'
    # lbl_name = '1mm_mask_no_dilate.npy'
    # csv_info = GetDataEDAfromCSV(save_csv_path)
    # data_list_all = [[Path(i['Path']).joinpath(img_name), Path(i['Path']).joinpath(lbl_name)] for i in csv_info]
    # K_Folder = KFoldSplit(data_list_all)
    # data_list_val = K_Folder[1]['val']
    # pred_name = 'label_pred.npy'
    # label_list = [i[1] for i in data_list_val]
    # pred_list = [i.parent.joinpath(pred_name) for i in label_list]
    # pred_result = TestThreshold(label_list, pred_list, threshold_list=[0.5, 0.6, 0.7, 0.8, 0.9])
