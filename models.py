import torch.nn as nn
import torch


def decorator(cls):
    print(6666666)
    return cls


class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        super(Upsample, self).__init__()
        self.align_corners = align_corners
        self.mode = mode
        self.scale_factor = scale_factor
        self.size = size

    def forward(self, x):
        return nn.functional.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode,
                                         align_corners=self.align_corners)


# @decorator
class Encoder(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size=3, downsample_rate=2, acf_func=nn.LeakyReLU):
        super(Encoder, self).__init__()
        self.block = nn.Sequential(nn.Conv3d(in_channels=in_chn,
                                             out_channels=out_chn,
                                             kernel_size=kernel_size,
                                             stride=downsample_rate,
                                             padding=1,
                                             bias=False),
                                   nn.InstanceNorm3d(num_features=out_chn),  # nn.LeakyReLU(inplace=True),
                                   acf_func(inplace=True),
                                   nn.Conv3d(in_channels=out_chn,
                                             out_channels=out_chn,
                                             kernel_size=kernel_size,
                                             stride=1,
                                             padding=1,
                                             bias=False),
                                   nn.InstanceNorm3d(num_features=out_chn),
                                   acf_func(inplace=True),  # nn.LeakyReLU(inplace=True),
                                   )

    def forward(self, x):
        x = self.block(x)
        return x


# @decorator
class Decoder(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size=3, acf_func=nn.LeakyReLU):
        super(Decoder, self).__init__()
        self.block = nn.Sequential(nn.ConvTranspose3d(in_channels=in_chn,
                                                      out_channels=out_chn,
                                                      kernel_size=2,
                                                      stride=2,
                                                      # padding=1,
                                                      bias=False),
                                   nn.InstanceNorm3d(num_features=out_chn),
                                   acf_func(inplace=True),  # nn.LeakyReLU(inplace=True),
                                   nn.Conv3d(in_channels=out_chn,
                                             out_channels=out_chn,
                                             kernel_size=kernel_size,
                                             stride=1,
                                             padding=1,
                                             bias=False),
                                   nn.InstanceNorm3d(num_features=out_chn),
                                   acf_func(inplace=True),  # nn.LeakyReLU(inplace=True),
                                   )

    def forward(self, x):
        x = self.block(x)
        return x


class OutputBlock(nn.Module):
    def __init__(self, in_chn, out_chn=1, kernel_size=3, downsample_rate=1, acf_func=nn.LeakyReLU):
        super(OutputBlock, self).__init__()
        if out_chn == 1:
            act_func2 = nn.Sigmoid()
        elif out_chn > 1:
            act_func2 = nn.Softmax(dim=1)
        else:
            raise Exception('Unexpected output channel {}'.format(out_chn))
        # self.block = nn.Sequential(nn.Conv3d(in_channels=in_chn,
        #                                      out_channels=out_chn,
        #                                      kernel_size=kernel_size,
        #                                      stride=downsample_rate,
        #                                      padding=1,
        #                                      bias=False),
        #                            nn.InstanceNorm3d(num_features=out_chn),
        #                            acf_func(inplace=True),  # nn.LeakyReLU(inplace=True),
        #                            nn.Conv3d(in_channels=out_chn,
        #                                      out_channels=out_chn,
        #                                      kernel_size=1,
        #                                      stride=1,
        #                                      padding=0,
        #                                      bias=True),
        #                            # nn.InstanceNorm3d(num_features=out_chn),
        #                            act_func2,  # nn.LeakyReLU(inplace=True),
        #                            )
        self.block = nn.Sequential(nn.Conv3d(in_channels=in_chn,
                                             out_channels=out_chn,
                                             kernel_size=1,
                                             stride=1,
                                             padding=0,
                                             bias=True),
                                   # nn.InstanceNorm3d(num_features=out_chn),
                                   act_func2,  # nn.LeakyReLU(inplace=True),
                                   )

    def forward(self, x):
        x = self.block(x)
        return x


class VanillaVNet(nn.Module):
    UPSAMPLE_MODE = 'trilinear'  #'bilinear'
    def __init__(self, downsample_times=5):
        super(VanillaVNet, self).__init__()
        self.e1 = Encoder(in_chn=1, out_chn=32, downsample_rate=1)
        self.e2 = Encoder(in_chn=32, out_chn=64)
        self.e3 = Encoder(in_chn=64, out_chn=128)
        self.e4 = Encoder(in_chn=128, out_chn=256)
        self.e5 = Encoder(in_chn=256, out_chn=320)
        self.e6 = Encoder(in_chn=320, out_chn=320)
        self.d6 = Decoder(in_chn=320, out_chn=320)
        self.d5 = Decoder(in_chn=320 * 2, out_chn=256)
        self.d4 = Decoder(in_chn=256 * 2, out_chn=128)
        self.d3 = Decoder(in_chn=128 * 2, out_chn=64)
        self.d2 = Decoder(in_chn=64 * 2, out_chn=32)
        # self.d1 = Encoder(in_chn=32 * 2, out_chn=1, downsample_rate=1)
        self.d1 = OutputBlock(in_chn=32 * 2, out_chn=1, downsample_rate=1)
        self.deepsupervision1 = nn.Sequential(nn.Conv3d(in_channels=64, out_channels=1, kernel_size=1, stride=1),
                                              nn.Sigmoid(),
                                              )
        self.deepsupervision2 = nn.Sequential(nn.Conv3d(in_channels=128, out_channels=1, kernel_size=1, stride=1),
                                              nn.Sigmoid(),
                                              )
        self.deepsupervision3 = nn.Sequential(nn.Conv3d(in_channels=256, out_channels=1, kernel_size=1, stride=1),
                                              nn.Sigmoid(),
                                              )


    def forward(self, x):
        x = self.e1(x)
        e2 = self.e2(x)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e6 = self.d6(e6)
        e6 = self.d5(torch.cat((e5, e6), 1))
        deepsupervision3 = self.deepsupervision3(e6)
        e6 = self.d4(torch.cat((e4, e6), 1))
        deepsupervision2 = self.deepsupervision2(e6)
        e6 = self.d3(torch.cat((e3, e6), 1))
        deepsupervision1 = self.deepsupervision1(e6)
        e6 = self.d2(torch.cat((e2, e6), 1))
        e6 = self.d1(torch.cat((x, e6), 1))
        return tuple([e6, deepsupervision1, deepsupervision2, deepsupervision3])


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """

    def __init__(self, smooth=1, p=1, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)
        num = 2 * torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth
        loss = 1 - num / den
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


if __name__ == '__main__':
    device = torch.device('cuda')
    a = torch.ones(1, 1, 128, 128, 128).to(device)
    m = VanillaVNet().to(device)
    criterion = BinaryDiceLoss()
    b, s1, s2, s3 = m(a)
    # loss = criterion(a, b)
