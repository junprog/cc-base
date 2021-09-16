import torch.nn as nn
import torch
import torch.nn.functional as F

from models.bagnet import BagNet
from models.resnet import ResNet

## bagnet + resnet
## bagnet + vgg

class BagResNet(nn.Module):
    def __init__(self, init_weight, bag_arch='bagnet33', res_arch='resnet50'):
        super(BagResNet, self).__init__()
        self.seen = 0

        self.relu = nn.ReLU(inplace=True)

        # BagNet [B x 3 x 512 x 512]
        self.bag_conv1, self.bag_conv2, self.bag_bn1, self.bag_layer1, self.bag_layer2, self.bag_layer3, self.bag_layer4 = self._make_baglayer(bag_arch)
        
        # ResNet
        self.res_conv, self.res_bn, self.res_pool, self.res_layer1, self.res_layer2, self.res_layer3, self.res_layer4 = self._make_reslayer(res_arch)
        
        # BagNet [B x 64 x 510 x 510]
        # self.block1 = Block([int(64*ratio), int(64*ratio), 'M'], first_block=True)
        # self.block2 = Block([int(128*ratio), int(128*ratio), 'M'], in_channels=int(64*ratio))
        # self.block3 = Block([int(256*ratio), int(256*ratio), int(256*ratio), 'M'], in_channels=int(128*ratio))
        # self.block4 = Block([int(512*ratio), int(512*ratio), int(512*ratio)], in_channels=int(256*ratio))

        self.backend_feat = [int(512*ratio), int(512*ratio), int(512*ratio), int(256*ratio), int(128*ratio), 64]
        self.backend = make_layers(self.backend_feat, in_channels=int(512*ratio), d_rate=2)

        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        if init_weight:
            self._initialize_weights(mode='normal')
        else:
            self._initialize_weights(mode='kaiming')
                
    def forward(self, x):


        RGB, T, shared = self.block1(RGB, T, None)
        RGB, T, shared = self.block2(RGB, T, shared)
        RGB, T, shared = self.block3(RGB, T, shared)
        _, _, shared = self.block4(RGB, T, shared)

        fusion_feature = shared

        fusion_feature = self.backend(fusion_feature)
        map = self.output_layer(fusion_feature)

        return map

    def _make_layer(self, block, planes, blocks, stride=1, kernel3=0, prefix=''):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                        kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        kernel = 1 if kernel3 == 0 else 3
        layers.append(block(self.inplanes, planes, stride, downsample, kernel_size=kernel))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            kernel = 1 if kernel3 <= i else 3
            layers.append(block(self.inplanes, planes, kernel_size=kernel))

        return nn.Sequential(*layers)

    def _make_baglayer(self, arch, pool_num=5):
        model = BagNet(arch=arch, pool_num=pool_num)

        bag_conv1 = model.feature_extracter.conv1
        bag_conv2 = model.feature_extracter.conv2
        bag_bn1 = model.feature_extracter.bn1
        
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

    def _initialize_weights(self, mode):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if mode == 'normal':
                    nn.init.normal_(m.weight, std=0.01)
                elif mode == 'kaiming':
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class Block(nn.Module):
    def __init__(self, cfg, in_channels=3, first_block=False, dilation_rate=1):
        super(Block, self).__init__()
        self.seen = 0
        self.first_block = first_block
        self.d_rate = dilation_rate

        if first_block is True:
            rgb_in_channels = 3
            t_in_channels = 1
        else:
            rgb_in_channels = in_channels
            t_in_channels = in_channels

        self.rgb_conv = make_layers(cfg, in_channels=rgb_in_channels, d_rate=self.d_rate)
        self.t_conv = make_layers(cfg, in_channels=t_in_channels, d_rate=self.d_rate)
        if first_block is False:
            self.shared_conv = make_layers(cfg, in_channels=in_channels, d_rate=self.d_rate)

        channels = cfg[0]
        self.rgb_msc = MSC(channels)
        self.t_msc = MSC(channels)
        if first_block is False:
            self.shared_fuse_msc = MSC(channels)
        self.shared_distribute_msc = MSC(channels)

        self.rgb_fuse_1x1conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.t_fuse_1x1conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.rgb_distribute_1x1conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.t_distribute_1x1conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, RGB, T, shared):
        RGB = self.rgb_conv(RGB)
        T = self.t_conv(T)
        if self.first_block:
            shared = torch.zeros(RGB.shape).cuda()
        else:
            shared = self.shared_conv(shared)

        new_RGB, new_T, new_shared = self.fuse(RGB, T, shared)
        return new_RGB, new_T, new_shared

    def fuse(self, RGB, T, shared):

        RGB_m = self.rgb_msc(RGB)
        T_m = self.t_msc(T)
        if self.first_block:
            shared_m = shared  # zero
        else:
            shared_m = self.shared_fuse_msc(shared)

        rgb_s = self.rgb_fuse_1x1conv(RGB_m - shared_m)
        rgb_fuse_gate = torch.sigmoid(rgb_s)
        t_s = self.t_fuse_1x1conv(T_m - shared_m)
        t_fuse_gate = torch.sigmoid(t_s)
        new_shared = shared + (RGB_m - shared_m) * rgb_fuse_gate + (T_m - shared_m) * t_fuse_gate

        new_shared_m = self.shared_distribute_msc(new_shared)
        s_rgb = self.rgb_distribute_1x1conv(new_shared_m - RGB_m)
        rgb_distribute_gate = torch.sigmoid(s_rgb)
        s_t = self.t_distribute_1x1conv(new_shared_m - T_m)
        t_distribute_gate = torch.sigmoid(s_t)
        new_RGB = RGB + (new_shared_m - RGB_m) * rgb_distribute_gate
        new_T = T + (new_shared_m - T_m) * t_distribute_gate

        return new_RGB, new_T, new_shared


class MSC(nn.Module):
    def __init__(self, channels):
        super(MSC, self).__init__()
        self.channels = channels
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=4)

        self.conv = nn.Sequential(
            nn.Conv2d(3*channels, channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = F.interpolate(self.pool1(x), x.shape[2:])
        x2 = F.interpolate(self.pool2(x), x.shape[2:])
        concat = torch.cat([x, x1, x2], 1)  # (1, 3C, H, W)
        fusion = self.conv(concat)

        return fusion


class IADM(nn.Module):
    def __init__(self, first_block=False):
        super(IADM, self).__init__()

        self.bagnet_block
        self.resnet_block

        if not first_block:
            self.shared_conv = nn.Conv2d


class Gate(nn.Module):
    def __init__(self, channels):
        super(Gate, self).__init__()
        self.conv1x1 = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        gate = torch.sigmoid(x)
        out = x * gate

        return out


class ScaleAdaptiveLayer(nn.Module):
    def __init__(self, target_shape, in_channels, out_channels):
        super(ScaleAdaptiveLayer, self).__init__()
        self.target_shape = target_shape

        self.scale_weight = nn.Parameter(torch.ones(out_channels))
        self.scale_bias = nn.Parameter(torch.zeros(out_channels))

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        _, c, w, h = x.size()
        if not self.target_shape == (w, h):
            x = F.interpolate(x, self.target_shape)
        x = x * self.scale_weight.reshape((1, c, 1, 1)) + self.scale_bias.reshape((1, c, 1, 1))
        out = self.conv(x)
        
        return out


def make_layers(cfg, in_channels=3, batch_norm=False, d_rate=False):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)