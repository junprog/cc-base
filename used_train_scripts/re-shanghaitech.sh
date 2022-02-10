#!bin/bash

## sta
python train.py --save-dir /mnt/hdd02/res-bagnet/layer-last/imagenet-pretrained --exp sta-resnet50-pretrained-400ep --arch resnet50 --pool-num 5 --pretrained --dataset shanghai-tech-a
python test.py --save-dir /mnt/hdd02/res-bagnet/layer-last/imagenet-pretrained/sta-resnet50-pretrained-400ep --arch resnet50 --pool-num 5 --dataset shanghai-tech-a

## stb
python train.py --save-dir /mnt/hdd02/res-bagnet/layer-last/imagenet-pretrained --exp stb-resnet50-pretrained-400ep --arch resnet50 --pool-num 5 --pretrained --dataset shanghai-tech-b --batch-size 4
python test.py --save-dir /mnt/hdd02/res-bagnet/layer-last/imagenet-pretrained/stb-resnet50-pretrained-400ep --arch resnet50 --pool-num 5 --dataset shanghai-tech-b