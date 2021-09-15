import torch.nn as nn
import torch
import torch.nn.functional as F

from bagnet import BagNet
from resnet import ResNet

## bagnet + resnet
## bagnet + vgg

class ScaleAdaptiveLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ScaleAdaptiveLayer, self).__init__()
        #self.scale_weight = nn.Parameter(torch.ones(out_channels))
        #self.scale_bias = nn.Parameter(torch.zeros(out_channels))

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, target_shape):
        _, c, w, h = x.size()
        if not target_shape == (w, h):
            x = F.interpolate(x, target_shape)
        #x = x * self.scale_weight.reshape((1, c, 1, 1)) + self.scale_bias.reshape((1, c, 1, 1))
        print(x.shape)
        out = self.conv(x)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)

        return out


class ShareLayer(nn.Module):
    def __init__(self, first_block, channels, num_blocks, stride, mode='concat'):
        super(ShareLayer, self).__init__()
        self.first_block = first_block
        self.in_planes = channels
        self.mode = mode

        self.bag_sc_layer = ScaleAdaptiveLayer(in_channels=channels*4, out_channels=channels*4)
        self.res_sc_layer = ScaleAdaptiveLayer(in_channels=channels*4, out_channels=channels*4)
        self.share_layer = self._make_layer(Bottleneck, channels, num_blocks, stride)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for i, stride in enumerate(strides):
            if i == 0:
                if self.mode == 'concat':
                    if self.first_block:
                        layers.append(block(2 * (self.in_planes*4), planes, stride))
                    else:
                        layers.append(block(2 * (self.in_planes*4) + (self.in_planes*2), planes, stride))
                elif self.mode == 'add':
                    layers.append(block(self.in_planes, planes, stride))
            else:
                layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, bag_x, res_x, share_x):
        target_shape = (int((bag_x.size(2) + res_x.size(2)) / 2), int((bag_x.size(3) + res_x.size(3)) / 2))

        bag_x = self.bag_sc_layer(bag_x, target_shape)
        res_x = self.res_sc_layer(res_x, target_shape)

        if self.mode == 'concat':
            if share_x is None:
                share_x = torch.cat([bag_x, res_x], dim=1)
            else:
                share_x = nn.AdaptiveAvgPool2d(target_shape)(share_x) ## down sampleで学習パラメータつける？
                share_x = torch.cat([bag_x, res_x, share_x], dim=1)
        elif self.mode == 'add':
            share_x = bag_x + res_x

        print(share_x.shape)
        share_x = self.share_layer(share_x)

        return share_x

class BagResNet(nn.Module):
    def __init__(self, bag_arch='bagnet33', res_arch='resnet50', pool_num=5):
        super(BagResNet, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        # BagNet
        self.bag_conv1, self.bag_conv2, self.bag_bn1, self.bag_layer1, self.bag_layer2, self.bag_layer3, self.bag_layer4 = self._make_baglayer(bag_arch, pool_num=pool_num)
        # ResNet
        self.res_conv, self.res_bn, self.res_pool, self.res_layer1, self.res_layer2, self.res_layer3, self.res_layer4 = self._make_reslayer(res_arch, pool_num=pool_num)

        self.share_layer1 = ShareLayer(first_block=True, channels=64, num_blocks=3, stride=1)
        self.share_layer2 = ShareLayer(first_block=False, channels=128, num_blocks=4, stride=1)
        self.share_layer3 = ShareLayer(first_block=False, channels=256, num_blocks=6, stride=1)
        self.share_layer4 = ShareLayer(first_block=False, channels=512, num_blocks=3, stride=1)

        self.regresser = make_regresser(pool_num)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
    
    def forward(self, x):
        # BagNet flow
        bag_x_0 = self.bag_conv1(x)
        bag_x_0 = self.bag_conv2(bag_x_0)
        bag_x_0 = self.bag_bn1(bag_x_0)
        bag_x_0 = self.relu(bag_x_0)            # [B, 64, 510, 510]

        bag_x_1 = self.bag_layer1(bag_x_0)      # [B, 256, 254, 254]
        bag_x_2 = self.bag_layer2(bag_x_1)      # [B, 512, 126, 126]
        bag_x_3 = self.bag_layer3(bag_x_2)      # [B, 1024, 62, 62]
        bag_x_4 = self.bag_layer4(bag_x_3)      # [B, 2048, 60, 60]

        # ResNet flow
        res_x_0 = self.res_conv(x)
        res_x_0 = self.res_bn(res_x_0)
        res_x_0 = self.relu(res_x_0)
        res_x_0 = self.res_pool(res_x_0)        # [B, 64, 256, 256]

        res_x_1 = self.res_layer1(res_x_0)      # [B, 256, 128, 128]
        res_x_2 = self.res_layer2(res_x_1)      # [B, 512, 64, 64]
        res_x_3 = self.res_layer3(res_x_2)      # [B, 1024, 32, 32]
        res_x_4 = self.res_layer4(res_x_3)      # [B, 2048, 16, 16]

        # Share flow
        share_x_1 = self.share_layer1(bag_x_1, res_x_1, None)
        share_x_2 = self.share_layer2(bag_x_2, res_x_2, share_x_1)
        share_x_3 = self.share_layer3(bag_x_3, res_x_3, share_x_2)
        share_x_4 = self.share_layer4(bag_x_4, res_x_4, share_x_3)

        x = self.regresser(share_x_4)
        x = self.output_layer(x)
        return torch.abs(x)

    def _make_baglayer(self, arch, pool_num=5):
        model = BagNet(arch=arch, pool_num=pool_num)

        bag_conv1 = model.feature_extracter.conv1
        bag_conv2 = model.feature_extracter.conv2
        bag_bn1 = model.feature_extracter.bn2d
        
        bag_layer1 = model.feature_extracter.block1
        bag_layer2 = model.feature_extracter.block2
        bag_layer3 = model.feature_extracter.block3
        bag_layer4 = model.feature_extracter.block4
        return bag_conv1, bag_conv2, bag_bn1, bag_layer1, bag_layer2, bag_layer3, bag_layer4

    def _make_reslayer(self, arch, pool_num=5):
        model = ResNet(arch=arch, pool_num=pool_num)

        res_conv = model.feature_extracter.conv2d
        res_bn = model.feature_extracter.bn2d
        res_pool = model.feature_extracter.maxpool
        
        res_layer1 = model.feature_extracter.layer1
        res_layer2 = model.feature_extracter.layer2
        res_layer3 = model.feature_extracter.layer3
        res_layer4 = model.feature_extracter.layer4

        return res_conv, res_bn, res_pool, res_layer1, res_layer2, res_layer3, res_layer4

def make_regresser(pool_num):
    base_ch = 64
    layers = []
    for i in range(pool_num, 0, -1):  
        conv2d = nn.Conv2d(base_ch*(2**i), base_ch*(2**(i-1)), kernel_size=3, padding=1)
        layers += [conv2d, nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)

if __name__ == '__main__':

    model = BagResNet()
    print(model)

    x = torch.rand(1, 3, 512, 512)
    out = model(x)

    print(out.shape)