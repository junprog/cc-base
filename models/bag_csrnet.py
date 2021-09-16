import torch
import torch.nn as nn
import torch.nn.functional as F

from models.bagnet import bagnet9, bagnet17, bagnet33

class BagCSRNet(nn.Module):
    def __init__(self, in_ch=3, arch='bagnet33', pool_num=5, up_scale=1, pretrained=False):
        super(BagCSRNet, self).__init__()
        """
        feature_extracter : ResNetの最終fc層なくした事前学習モデル
        regresser : channel数を削減する (regressiion)
        """
        if pool_num == 5:
            reg_in_ch = 2048
        elif pool_num == 4:
            reg_in_ch = 1024
        elif pool_num == 3:
            reg_in_ch = 512
        
        backend_feat  = [512, 512, 512, 256, 128, 64]

        self.feature_extracter = make_bagnet_feature_extracter(arch, pool_num, pretrained=pretrained)
        self.regresser = make_csrnet_regresser(backend_feat, in_channels=reg_in_ch, dilation=True)

        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        self.up_scale = up_scale

    def forward(self, x):
        x = self.feature_extracter(x)

        if self.up_scale != 1:
            x = F.interpolate(x, scale_factor=self.up_scale, mode='bilinear', align_corners=False)

        x = self.regresser(x)
        x = self.output_layer(x)

        return torch.abs(x)

def make_bagnet_feature_extracter(arch, pool_num, pretrained):
    if arch == 'bagnet33':
        model = bagnet33(pretrained=pretrained)
    elif arch == 'bagnet17':
        model = bagnet17(pretrained=pretrained)
    elif arch == 'bagnet9':
        model = bagnet9(pretrained=pretrained)

    layers = list(model.children())[:-2]

    extracter = nn.Sequential()
    for i in range(0, pool_num):
        if i == 0:
            extracter.add_module('conv1',layers[0])
            extracter.add_module('conv2',layers[1])
            extracter.add_module('bn2d',layers[2])
            extracter.add_module('relu',layers[3])
        else:
            extracter.add_module('block{}'.format(i),layers[i+3])

    return extracter

def make_csrnet_regresser(cfg, in_channels=3, batch_norm=False, dilation=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)  