#!bin/bash

# train
#python train.py --save-dir /mnt/hdd02/res-bagnet --exp stb-bagnet33-pretrained --arch bagnet33 --pool-num 5 --pretrained --dataset shanghai-tech-b --batch-size 2
#python train.py --save-dir /mnt/hdd02/res-bagnet --exp stb-resnet50-pretrained --arch resnet50 --pool-num 5 --pretrained --dataset shanghai-tech-b --batch-size 4
# python train.py --save-dir /mnt/hdd02/res-bagnet --exp stb-vgg19-pretrained --arch vgg19 --pool-num 4 --pretrained --dataset shanghai-tech-b
# python train.py --save-dir /mnt/hdd02/res-bagnet --exp stb-vgg19_bn-pretrained --arch vgg19_bn --pool-num 4 --pretrained --dataset shanghai-tech-b

#python train.py --save-dir /mnt/hdd02/res-bagnet --exp sta-bagnet33-pretrained --arch bagnet33 --pool-num 5 --pretrained --dataset shanghai-tech-a
#python train.py --save-dir /mnt/hdd02/res-bagnet --exp sta-resnet50-pretrained --arch resnet50 --pool-num 5 --pretrained --dataset shanghai-tech-a
# python train.py --save-dir /mnt/hdd02/res-bagnet --exp sta-vgg19-pretrained --arch vgg19 --pool-num 4 --pretrained --dataset shanghai-tech-a
# python train.py --save-dir /mnt/hdd02/res-bagnet --exp sta-vgg19_bn-pretrained --arch vgg19_bn --pool-num 4 --pretrained --dataset shanghai-tech-a

python train.py --save-dir /mnt/hdd02/res-bagnet --exp re_qnrf-vgg19-pretrained --arch vgg19 --pool-num 4 --pretrained --dataset rescale-ucf-qnrf
python train.py --save-dir /mnt/hdd02/res-bagnet --exp re_qnrf-vgg19_bn-pretrained --arch vgg19_bn --pool-num 4 --pretrained --dataset rescale-ucf-qnrf
#python train.py --save-dir /mnt/hdd02/res-bagnet --exp re_qnrf-bagnet33-pretrained --arch bagnet33 --pool-num 5 --pretrained --dataset rescale-ucf-qnrf
#python train.py --save-dir /mnt/hdd02/res-bagnet --exp re_qnrf-resnet50-pretrained --arch resnet50 --pool-num 5 --pretrained --dataset rescale-ucf-qnrf
# モデルでかすぎてVal通らない python train.py --save-dir /mnt/hdd02/res-bagnet --exp re_qnrf-fusionnet-pretrained --arch fusionnet --pool-num 5 --pretrained --dataset rescale-ucf-qnrf

# test
#python test.py --save-dir /mnt/hdd02/res-bagnet/stb-bagnet33-pretrained --arch bagnet33 --pool-num 5 --dataset shanghai-tech-b
#python test.py --save-dir /mnt/hdd02/res-bagnet/stb-resnet50-pretrained --arch resnet50 --pool-num 5 --dataset shanghai-tech-b
# python test.py --save-dir /mnt/hdd02/res-bagnet/stb-vgg19-pretrained --arch vgg19 --pool-num 4 --dataset shanghai-tech-b
# python test.py --save-dir /mnt/hdd02/res-bagnet/stb-vgg19_bn-pretrained --arch vgg19_bn --pool-num 4 --dataset shanghai-tech-b

#python test.py --save-dir /mnt/hdd02/res-bagnet/sta-bagnet33-pretrained --arch bagnet33 --pool-num 5 --dataset shanghai-tech-a
#python test.py --save-dir /mnt/hdd02/res-bagnet/sta-resnet50-pretrained --arch resnet50 --pool-num 5 --dataset shanghai-tech-a
# python test.py --save-dir /mnt/hdd02/res-bagnet/sta-vgg19-pretrained --arch vgg19 --pool-num 4 --dataset shanghai-tech-a
# python test.py --save-dir /mnt/hdd02/res-bagnet/sta-vgg19_bn-pretrained --arch vgg19_bn --pool-num 4 --dataset shanghai-tech-a

python test.py --save-dir /mnt/hdd02/res-bagnet/re_qnrf-vgg19-pretrained --arch vgg19 --pool-num 4 --dataset rescale-ucf-qnrf
python test.py --save-dir /mnt/hdd02/res-bagnet/re_qnrf-vgg19_bn-pretrained --arch vgg19_bn --pool-num 4 --dataset rescale-ucf-qnrf
#python test.py --save-dir /mnt/hdd02/res-bagnet/re_qnrf-bagnet33-pretrained --arch bagnet33 --pool-num 5 --dataset rescale-ucf-qnrf
#python test.py --save-dir /mnt/hdd02/res-bagnet/re_qnrf-resnet50-pretrained --arch resnet50 --pool-num 5 --dataset rescale-ucf-qnrf
# モデルでかすぎてVal通らない python test.py --save-dir /mnt/hdd02/res-bagnet/re_qnrf-fusionnet-pretrained --arch fusionnet --pool-num 5 --dataset rescale-ucf-qnrf