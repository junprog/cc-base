import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from collections import OrderedDict

class VGG(nn.Module):
    def __init__(self, in_ch=3, pool_num=3, model='vgg19_bn', up_scale=8, pretrain=False):
        super(VGG, self).__init__()
        """
        feature_extracter : VGGの最終fc層なくした事前学習モデル
        down_channels : channel数を削減する (regressiion)
        """
        if model == 'vgg19':
            self.feature_extracter = make_vgg19_feature_extracter(pool_num, in_ch=in_ch, bn=False, pretrain=pretrain)
            self.resgresser = make_vgg_down_channels()
        elif model == 'vgg19_bn':
            self.feature_extracter = make_vgg19_feature_extracter(pool_num, in_ch=in_ch, bn=True, pretrain=pretrain)
            self.resgresser = make_vgg_down_channels()

        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        self.up_scale = up_scale

    def forward(self, x):
        x = self.feature_extracter(x)

        if self.up_scale != 1:
            x = F.interpolate(x, scale_factor=self.up_scale, mode='bilinear', align_corners=False)

        x = self.resgresser(x)
        x = self.output_layer(x)

        return torch.abs(x)


def make_vgg19_feature_extracter(pool_num, in_ch=3, bn=False, pretrain=False):
    if bn == False:
        model = models.vgg19(pretrained=pretrain)
        head_flag = False
        if in_ch != 3:
            head_flag = True
            head_conv = nn.Conv2d(in_channels=in_ch, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        
        if pool_num == 3:
            layers = list(model.features.children())[:-10]
        elif pool_num == 4:
            layers = list(model.features.children())[:-1]

        if head_flag:
            layers[0] = head_conv
    else:
        model = models.vgg19_bn(pretrained=pretrain)
        head_flag = False
        if in_ch != 3:
            head_flag = True
            head_conv = nn.Conv2d(in_channels=in_ch, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        
        if pool_num == 3:
            layers = list(model.features.children())[:-14]
        elif pool_num == 4:
            layers = list(model.features.children())[:-1]

        if head_flag:
            layers[0] = head_conv

    return nn.Sequential(*layers)

def make_vgg_down_channels():
    base_ch = 64
    layers = []

    for i in range(3, 0, -1):
        conv2d = nn.Conv2d(base_ch*(2**(i)), base_ch*(2**(i-1)), kernel_size=3, padding=1)
        layers += [conv2d, nn.ReLU(inplace=True)]
        
    return nn.Sequential(*layers)