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

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True  # pre GPU optimization
    save_csv_path = Path(r'data.csv')
    img_name = '1mm.npy'
    lbl_name = '1mm_mask.npy'
    epoch_total = 20000
    writer = SummaryWriter('test_log')
    a = GetDataEDAfromCSV(save_csv_path)
    data_list = [[Path(i['Path']).joinpath(img_name), Path(i['Path']).joinpath(lbl_name)] for i in a]
    data_list = [data_list[5]]
    test_dataset = CoarseResolutionDataset(data_list, is_train=False)

    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=1,
                                 shuffle=True,
                                 num_workers=1)
    device = torch.device('cuda')
    net = models.VanillaVNet().to(device)
    # net = nnunet.Generic_UNet(input_channels=1, base_num_features=nnunet.Generic_UNet.BASE_NUM_FEATURES_3D,
    #                           num_classes=1, num_pool=5, num_conv_per_stage=2,
    #                           feat_map_mul_on_downscale=2, conv_op=nn.Conv3d,
    #                           norm_op=nn.InstanceNorm3d, norm_op_kwargs=None,
    #                           dropout_op=nn.Dropout3d, dropout_op_kwargs=None,
    #                           nonlin=nn.LeakyReLU, nonlin_kwargs=None, deep_supervision=True,
    #                           dropout_in_localization=False,
    #                           final_nonlin=nn.Sigmoid(), weightInitializer=nnunet.InitWeights_He(1e-2),
    #                           pool_op_kernel_sizes=None,
    #                           conv_kernel_sizes=None,
    #                           upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
    #                           max_num_features=None, basic_block=nnunet.ConvDropoutNormNonlin,
    #                           seg_output_use_bias=False)
    # net = net.to(device)
    criterion_DICE = models.BinaryDiceLoss()
    criterion_CE = nn.BCELoss()
    optim = torch.optim.Adam(net.parameters(),
                             lr=0.01,  # 3e-4,
                             betas=(0.9, 0.999),
                             weight_decay=0)
    for epoch in range(epoch_total):
        # plt.figure()
        loss_CE_last_item = 0
        loss_CE_loss_supervision1_item = 0
        loss_CE_loss_supervision2_item = 0
        loss_CE_loss_supervision3_item = 0
        loss_DICE_last_item = 0
        loss_DICE_loss_supervision1_item = 0
        loss_DICE_loss_supervision2_item = 0
        loss_DICE_loss_supervision3_item = 0
        print(epoch)
        for iter_, (x, y, y2, y4, y8) in enumerate(test_dataloader):
            print(x.shape)
            x = x.to(device).unsqueeze(1).to(device)
            y = y.to(device).unsqueeze(1).float().to(device)
            y2 = y2.to(device).unsqueeze(1).float().to(device)
            y4 = y4.to(device).unsqueeze(1).float().to(device)
            y8 = y8.to(device).unsqueeze(1).float().to(device)
            # x = x[:, :128, :128, :128].to(device).unsqueeze(1).to(device)
            # y = y[:, :128, :128, :128].to(device).unsqueeze(1).float().to(device)
            # x = x[:, :128, :128, :128].to(device).unsqueeze(1).to(device)
            # y = y[:, :128, :128, :128].to(device).unsqueeze(1).float().to(device)
            pred, supervision1, supervision2, supervision3 = net(x)
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
            loss_ce_item = loss_CE.detach().item()
            loss_dice_item = loss_DICE.detach().item()
            print('LOSS CE {}'.format(loss_ce_item))
            print('LOSS DICE {}'.format(loss_dice_item))
            loss_CE_last_item += loss_CE_last.detach().item()
            loss_CE_loss_supervision1_item += loss_CE_supervision1.detach().item()
            loss_CE_loss_supervision2_item += loss_CE_supervision2.detach().item()
            loss_CE_loss_supervision3_item += loss_CE_supervision3.detach().item()
            loss_DICE_last_item += loss_DICE_last.detach().item()
            loss_DICE_loss_supervision1_item += loss_DICE_supervision1.detach().item()
            loss_DICE_loss_supervision2_item += loss_DICE_supervision2.detach().item()
            loss_DICE_loss_supervision3_item += loss_DICE_supervision3.detach().item()
        writer.add_scalars('loss_all',
                           {'training_loss_all': loss_ce_item + loss_dice_item},
                           epoch)
        writer.add_scalars('loss_ce',
                           {'training_loss_last': loss_CE_last_item,
                            'training_loss_s1': loss_CE_loss_supervision1_item,
                            'training_loss_s2': loss_CE_loss_supervision2_item,
                            'training_loss_s3': loss_CE_loss_supervision3_item},
                           epoch)
        writer.add_scalars('loss_dice',
                           {'training_loss_dice_last': loss_DICE_last_item,
                            'training_loss_dice_s1': loss_DICE_loss_supervision1_item,
                            'training_loss_dice_s2': loss_DICE_loss_supervision2_item,
                            'training_loss_dice_s3': loss_DICE_loss_supervision3_item},
                           epoch)
        if epoch % 50 == 0:
            supervision1 = supervision1[0, 0, :, :, :].detach().to('cpu').numpy()
            supervision2 = supervision2[0, 0, :, :, :].detach().to('cpu').numpy()
            supervision3 = supervision3[0, 0, :, :, :].detach().to('cpu').numpy()
            supervision1 = scipy.ndimage.interpolation.zoom(supervision1, 2, mode='nearest', order=0)
            supervision2 = scipy.ndimage.interpolation.zoom(supervision2, 4, mode='nearest', order=0)
            supervision3 = scipy.ndimage.interpolation.zoom(supervision3, 8, mode='nearest', order=0)
            pred110 = pred[0, :, 110, :, :].to('cpu').detach()
            supervision1_110 = supervision1[np.newaxis, 110, :, :]
            supervision2_110 = supervision2[np.newaxis, 110, :, :]
            supervision3_110 = supervision3[np.newaxis, 110, :, :]
            y110 = y[0, :, 110, :, :].to('cpu').detach()
            pred109 = pred[0, :, 109, :, :].to('cpu').detach()
            supervision1_109 = supervision1[np.newaxis, 109, :, :]
            supervision2_109 = supervision2[np.newaxis, 109, :, :]
            supervision3_109 = supervision3[np.newaxis, 109, :, :]
            y109 = y[0, :, 109, :, :].to('cpu').detach()
            writer.add_image('pred110',
                             pred110, epoch // 50)
            writer.add_image('pred110_s1',
                             supervision1_110, epoch // 50)
            writer.add_image('pred110',
                             supervision2_110, epoch // 50)
            writer.add_image('pred110',
                             supervision3_110, epoch // 50)
            writer.add_image('y110',
                             y110, epoch // 50)
            writer.add_image('pred109',
                             pred109, epoch // 50)
            writer.add_image('pred109_s1',
                             supervision1_109, epoch // 50)
            writer.add_image('pred109_s2',
                             supervision2_109, epoch // 50)
            writer.add_image('pred109_s3',
                             supervision3_109, epoch // 50)
            writer.add_image('y109',
                             y109, epoch // 50)
            for tag, value in net.named_parameters():
                tag = tag.replace('.', '/')
                print(value.data.cpu().numpy().shape)
                # print(value.grad.data.cpu().numpy().shape)

    print('hello')
