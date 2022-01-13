#!bin/bash

# train
# python train.py --save-dir /mnt/hdd02/res-bagnet/layer-last-1 --exp stb-bagnet33-pretrained --arch bagnet33 --pool-num 4 --pretrained --dataset shanghai-tech-b --batch-size 2
# python train.py --save-dir /mnt/hdd02/res-bagnet/layer-last-1 --exp stb-resnet50-pretrained --arch resnet50 --pool-num 4 --pretrained --dataset shanghai-tech-b --batch-size 4
# python train.py --save-dir /mnt/hdd02/res-bagnet/layer-last-1 --exp stb-vgg19-pretrained --arch vgg19 --pool-num 3 --pretrained --dataset shanghai-tech-b
# python train.py --save-dir /mnt/hdd02/res-bagnet/layer-last-1 --exp stb-vgg19_bn-pretrained --arch vgg19_bn --pool-num 3 --pretrained --dataset shanghai-tech-b

python train.py --save-dir /mnt/hdd02/res-bagnet/layer-last-1 --exp sta-bagnet33-pretrained --arch bagnet33 --pool-num 4 --pretrained --dataset shanghai-tech-a
# python train.py --save-dir /mnt/hdd02/res-bagnet/layer-last-1 --exp sta-resnet50-pretrained --arch resnet50 --pool-num 4 --pretrained --dataset shanghai-tech-a
# python train.py --save-dir /mnt/hdd02/res-bagnet/layer-last-1 --exp sta-vgg19-pretrained --arch vgg19 --pool-num 3 --pretrained --dataset shanghai-tech-a
# python train.py --save-dir /mnt/hdd02/res-bagnet/layer-last-1 --exp sta-vgg19_bn-pretrained --arch vgg19_bn --pool-num 3 --pretrained --dataset shanghai-tech-a

python train.py --save-dir /mnt/hdd02/res-bagnet/layer-last-1 --exp re_qnrf-vgg19-pretrained --arch vgg19 --pool-num 3 --pretrained --dataset rescale-ucf-qnrf
#python train.py --save-dir /mnt/hdd02/res-bagnet/layer-last-1 --exp re_qnrf-vgg19_bn-pretrained --arch vgg19_bn --pool-num 3 --pretrained --dataset rescale-ucf-qnrf
python train.py --save-dir /mnt/hdd02/res-bagnet/layer-last-1 --exp re_qnrf-bagnet33-pretrained --arch bagnet33 --pool-num 4 --pretrained --dataset rescale-ucf-qnrf
# python train.py --save-dir /mnt/hdd02/res-bagnet/layer-last-1 --exp re_qnrf-resnet50-pretrained --arch resnet50 --pool-num 4 --pretrained --dataset rescale-ucf-qnrf
# モデルでかすぎてVal通らない python train.py --save-dir /mnt/hdd02/res-bagnet/layer-last-1 --exp re_qnrf-fusionnet-pretrained --arch fusionnet --pool-num 5 --pretrained --dataset rescale-ucf-qnrf

# test
python test.py --save-dir /mnt/hdd02/res-bagnet/layer-last-1/stb-bagnet33-pretrained --arch bagnet33 --pool-num 4 --dataset shanghai-tech-b
python test.py --save-dir /mnt/hdd02/res-bagnet/layer-last-1/stb-resnet50-pretrained --arch resnet50 --pool-num 4 --dataset shanghai-tech-b
python test.py --save-dir /mnt/hdd02/res-bagnet/layer-last-1/stb-vgg19-pretrained --arch vgg19 --pool-num 3 --dataset shanghai-tech-b
python test.py --save-dir /mnt/hdd02/res-bagnet/layer-last-1/stb-vgg19_bn-pretrained --arch vgg19_bn --pool-num 3 --dataset shanghai-tech-b

python test.py --save-dir /mnt/hdd02/res-bagnet/layer-last-1/sta-bagnet33-pretrained --arch bagnet33 --pool-num 4 --dataset shanghai-tech-a
python test.py --save-dir /mnt/hdd02/res-bagnet/layer-last-1/sta-resnet50-pretrained --arch resnet50 --pool-num 4 --dataset shanghai-tech-a
python test.py --save-dir /mnt/hdd02/res-bagnet/layer-last-1/sta-vgg19-pretrained --arch vgg19 --pool-num 3 --dataset shanghai-tech-a
python test.py --save-dir /mnt/hdd02/res-bagnet/layer-last-1/sta-vgg19_bn-pretrained --arch vgg19_bn --pool-num 3 --dataset shanghai-tech-a

python test.py --save-dir /mnt/hdd02/res-bagnet/layer-last-1/re_qnrf-vgg19-pretrained --arch vgg19 --pool-num 3 --dataset rescale-ucf-qnrf
python test.py --save-dir /mnt/hdd02/res-bagnet/layer-last-1/re_qnrf-vgg19_bn-pretrained --arch vgg19_bn --pool-num 3 --dataset rescale-ucf-qnrf
python test.py --save-dir /mnt/hdd02/res-bagnet/layer-last-1/re_qnrf-bagnet33-pretrained --arch bagnet33 --pool-num 4 --dataset rescale-ucf-qnrf
python test.py --save-dir /mnt/hdd02/res-bagnet/layer-last-1/re_qnrf-resnet50-pretrained --arch resnet50 --pool-num 4 --dataset rescale-ucf-qnrf
# モデルでかすぎてVal通らない python test.py --save-dir /mnt/hdd02/res-bagnet/layer-last-1/re_qnrf-fusionnet-pretrained --arch fusionnet --pool-num 5 --dataset rescale-ucf-qnrf