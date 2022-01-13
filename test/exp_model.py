import sys
import os

from torchvision.models import resnet
current_path = os.getcwd()
sys.path.append(current_path) # カレントディレクトリをパスに追加

import torch
from models.bagnet import BagNet
from models.resnet import ResNet
from models.fusion_model import BagResNet
from torchsummary import summary
from pthflops import count_ops


inputs = torch.rand(1, 3, 512, 512).cuda()
# model = BagNet(in_ch=3, arch='bagnet33', pool_num=5).cuda()
# model = ResNet(in_ch=3, arch='resnet50', pool_num=5).cuda()
model = BagResNet(pool_num=5).cuda()

count_ops(model, inputs)

# summary(model, (3, 512, 512))

# outputs = model(inputs, inputs)

# print(model)
# print(outputs.shape)