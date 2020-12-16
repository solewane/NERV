import torch
import torch.nn as nn
import models
from utils_dataset import CoarseResolutionDataset
from pathlib import Path
from torch.utils.data import DataLoader
from utils import GetDataEDAfromCSV
from torch.utils.tensorboard import SummaryWriter
import nnunet
import scipy.ndimage
import numpy as np
import random
from sklearn.model_selection import KFold
from tqdm import tqdm
import time


def ThrowDice(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子


def CheckpointSave(model, optimizer, epoch,  path):
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


def Train_net(net,
              epochs=2000,
              random_seed=24,
              ):
    pass


if __name__ == '__main__':
    checkpoint_flag = False
    checkpoint_model = None
    ThrowDice(24)
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True  # pre GPU optimization
    save_csv_path = Path(r'data.csv')
    img_name = '1mm.npy'
    lbl_name = '1mm_mask_no_dilate.npy'
    epoch_total = 3000
    init_lr = 0.01
    save_freq = 500
    writer = SummaryWriter('test_log')
    csv_info = GetDataEDAfromCSV(save_csv_path)
    data_list_all = [[Path(i['Path']).joinpath(img_name), Path(i['Path']).joinpath(lbl_name)] for i in csv_info]
    K_Folder = KFoldSplit(data_list_all)
    data_list = K_Folder[0]['train']  # k folder
    data_list_val = K_Folder[0]['val']
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
                                            is_morphology=True,
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
                              dropout_op=nn.Dropout3d, dropout_op_kwargs=None,
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
                loss_DICE = loss_DICE_last + 0.5 * loss_DICE_supervision1 + 0.25 * loss_DICE_supervision2 + 0.125 * loss_DICE_supervision3
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
        net.eval()
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
                loss_ce_item_val += loss_CE.detach().item() * n_batch / n_train
                loss_dice_item_val += loss_DICE.detach().item() * n_batch / n_train
                # print('LOSS CE {}'.format(loss_ce_item))
                # print('LOSS DICE {}'.format(loss_dice_item))
                loss_CE_last_item_val += loss_CE_last.detach().item() * n_batch / n_train
                loss_CE_loss_supervision1_item_val += loss_CE_supervision1.detach().item() * n_batch / n_train
                loss_CE_loss_supervision2_item_val += loss_CE_supervision2.detach().item() * n_batch / n_train
                loss_CE_loss_supervision3_item_val += loss_CE_supervision3.detach().item() * n_batch / n_train
                loss_DICE_last_item_val += loss_DICE_last.detach().item() * n_batch / n_train
                loss_DICE_loss_supervision1_item_val += loss_DICE_supervision1.detach().item() * n_batch / n_train
                loss_DICE_loss_supervision2_item_val += loss_DICE_supervision2.detach().item() * n_batch / n_train
                loss_DICE_loss_supervision3_item_val += loss_DICE_supervision3.detach().item() * n_batch / n_train
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
        if epoch % 50 == 0:
            # path = 'save/epoch{}_CE{:.4f}_Dice{:.4f}.pth'.format(epoch, loss_CE_last_item, loss_DICE_last_item)
            # CheckpointSave(net, optim, epoch, path)
            path = 'save/epoch{}_CE{:.4f}_Dice{:.4f}.model'.format(epoch, loss_CE_last_item, loss_DICE_last_item)
            Save_Checkpoint(net, epoch, path)
    print('hello')
