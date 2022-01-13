import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import math

from collections import OrderedDict

class ResNet(nn.Module):
    def __init__(self, in_ch=3, arch='resnet50', pool_num=5, up_scale=1, pretrained=False, feat_freeze=False):
        super(ResNet, self).__init__()
        """
        feature_extracter : ResNetの最終fc層なくした事前学習モデル
        regresser : channel数を削減する (regressiion)
        """

        self.feature_extracter = make_resnet_feature_extracter(arch, pool_num, in_ch=in_ch, pretrained=pretrained)
        self.regresser = make_resnet_regresser(arch, pool_num)

        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        self.up_scale = up_scale

        if feat_freeze:
            for params in self.feature_extracter.parameters():
                params.requires_grad = False

    def forward(self, x):
        x = self.feature_extracter(x)

        if self.up_scale != 1:
            x = F.interpolate(x, scale_factor=self.up_scale, mode='bilinear', align_corners=False)

        x = self.regresser(x)
        x = self.output_layer(x)

        return torch.abs(x)


def make_resnet_feature_extracter(arch, pool_num, in_ch=3, pretrained=False):
    if arch == 'resnet18' or arch == 'resnet34':
        if arch == 'resnet18':
            model = models.resnet18(pretraineded=pretrained)
        if arch == 'resnet34':
            model = models.resnet34(pretraineded=pretrained)

        layers = list(model.children())[:-2]

        extracter = nn.Sequential()
        for i in range(0, pool_num):
            if i == 0:
                extracter.add_module('conv2d',layers[0])
                extracter.add_module('bn2d',layers[1])
                extracter.add_module('relu',layers[2])
                extracter.add_module('maxpool',layers[3])
            else:
                extracter.add_module('layer{}'.format(i),layers[i+3])

    else:
        if arch == 'resnet50':
            model = models.resnet50(pretrained=pretrained)
        elif arch == 'resnet101':
            model = models.resnet101(pretrained=pretrained)

        layers = list(model.children())[:-2]

        extracter = nn.Sequential()

        head_flag = False
        if in_ch != 3:
            head_flag = True
            head_conv = nn.Conv2d(in_channels=in_ch, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        for i in range(0, pool_num):
            if i == 0:
                if head_flag:
                    extracter.add_module('conv2d', head_conv)
                else:
                    extracter.add_module('conv2d',layers[0])
                extracter.add_module('bn2d',layers[1])
                extracter.add_module('relu',layers[2])
                extracter.add_module('maxpool',layers[3])
            else:
                extracter.add_module('layer{}'.format(i),layers[i+3])

    return extracter

def make_resnet_regresser(arch, pool_num):
    if arch == 'resnet18' or arch == 'resnet34':
        base_ch = 64
        layers = []

        for i in range(pool_num, 2, -1):  
            conv2d = nn.Conv2d(base_ch*(2**(i-2)), base_ch*(2**(i-3)), kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]

    else:
        base_ch = 64
        layers = []

        for i in range(pool_num, 0, -1):  
            conv2d = nn.Conv2d(base_ch*(2**i), base_ch*(2**(i-1)), kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
        
    return nn.Sequential(*layers)