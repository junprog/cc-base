import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from collections import OrderedDict

class VGG_BagNet(nn.Module):
    def __init__(self, in_ch=3, arch='vgg19_bn', pool_num=4, up_scale=1, pretrained=False):
        super(VGG_BagNet, self).__init__()
        """
        feature_extracter : VGGの最終fc層なくした事前学習モデル
        regresser : channel数を削減する (regressiion)
        """
        if arch == 'vgg19_bag':
            self.feature_extracter = make_vgg19_bagnet_feature_extracter(pool_num, in_ch=in_ch, bn=False, pretrained=pretrained)
            self.regresser = make_vgg_regresser()
        elif arch == 'vgg19_bag_bn':
            self.feature_extracter = make_vgg19_bagnet_feature_extracter(pool_num, in_ch=in_ch, bn=True, pretrained=pretrained)
            self.regresser = make_vgg_regresser()

        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        self.up_scale = up_scale

    def forward(self, x):
        x = self.feature_extracter(x)

        if self.up_scale != 1:
            x = F.interpolate(x, scale_factor=self.up_scale, mode='bilinear', align_corners=False)

        x = self.regresser(x)
        x = self.output_layer(x)

        return torch.abs(x)

def make_vgg19_bagnet_feature_extracter(pool_num, in_ch=3, bn=False, pretrained=False):
    if bn == False:
        model = models.vgg19(pretrained=pretrained)
        head_flag = False
        if in_ch != 3:
            head_flag = True
            head_conv = nn.Conv2d(in_channels=in_ch, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        
        if pool_num == 4:
            layers = list(model.features.children())[:-10]
        elif pool_num == 5:
            layers = list(model.features.children())[:-1]

        if head_flag:
            layers[0] = head_conv
    else:
        model = vgg19_bag_bn(pretrained=False)
        model.load_state_dict(torch.load('models/weights/vgg19_bag_bn_best.pth.tar')['state_dict'], strict=False)
        head_flag = False
        if in_ch != 3:
            head_flag = True
            head_conv = nn.Conv2d(in_channels=in_ch, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        
        if pool_num == 4:
            layers = list(model.features.children())[:-14]
        elif pool_num == 5:
            layers = list(model.features.children())[:-1]

        if head_flag:
            layers[0] = head_conv

    return nn.Sequential(*layers)

def make_vgg_regresser():
    base_ch = 64
    layers = []

    for i in range(3, 0, -1):
        conv2d = nn.Conv2d(base_ch*(2**(i)), base_ch*(2**(i-1)), kernel_size=3, padding=1)
        layers += [conv2d, nn.ReLU(inplace=True)]
        
    return nn.Sequential(*layers)


### Classification Model ##
from typing import Union, List, Dict, Any, cast
from collections import OrderedDict

class _VGGBag_cls(nn.Module):
    
    def __init__(
        self,
        features: nn.Module,
        num_classes: int = 1000,
        init_weights: bool = True
    ) -> None:

        super(_VGGBag_cls, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        print("kaming init done")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3

    for i, v in enumerate(cfg):
        v = cast(int, v)

        if(i == 0):
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, stride=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
        elif((i == 2) or (i == 4) or (i == 8) or (i == 12)):
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, stride=2)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=1, stride=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]

        in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    'E': [64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512],
}


def _vgg_bag(arch: str, cfg: str, batch_norm: bool, pretrained: bool, progress: bool, **kwargs: Any) -> _VGGBag_cls:

    """
    if pretrained:
        kwargs['init_weights'] = False
    """

    model = _VGGBag_cls(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    return model


def vgg19_bag_bn(pretrained: bool = False, progress: bool = True, stop_imagenet_require_grad: bool = False, **kwargs: Any) -> _VGGBag_cls:
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg_bag('vgg19_bn', 'E', True, pretrained, progress, **kwargs)