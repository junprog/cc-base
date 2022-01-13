#!bin/bash

# ↓↓↓↓ from scratch ↓↓↓↓

# train
python train.py --save-dir /mnt/hdd02/res-bagnet/layer-last/scratch --exp stb-bagnet33-scratch --arch bagnet33 --pool-num 5  --dataset shanghai-tech-b --batch-size 2
python train.py --save-dir /mnt/hdd02/res-bagnet/layer-last/scratch --exp stb-resnet50-scratch --arch resnet50 --pool-num 5  --dataset shanghai-tech-b --batch-size 4
python train.py --save-dir /mnt/hdd02/res-bagnet/layer-last/scratch --exp stb-vgg19-scratch --arch vgg19 --pool-num 4  --dataset shanghai-tech-b
python train.py --save-dir /mnt/hdd02/res-bagnet/layer-last/scratch --exp stb-vgg19_bn-scratch --arch vgg19_bn --pool-num 4  --dataset shanghai-tech-b

python train.py --save-dir /mnt/hdd02/res-bagnet/layer-last/scratch --exp sta-bagnet33-scratch --arch bagnet33 --pool-num 5  --dataset shanghai-tech-a
python train.py --save-dir /mnt/hdd02/res-bagnet/layer-last/scratch --exp sta-resnet50-scratch --arch resnet50 --pool-num 5  --dataset shanghai-tech-a
python train.py --save-dir /mnt/hdd02/res-bagnet/layer-last/scratch --exp sta-vgg19-scratch --arch vgg19 --pool-num 4  --dataset shanghai-tech-a
python train.py --save-dir /mnt/hdd02/res-bagnet/layer-last/scratch --exp sta-vgg19_bn-scratch --arch vgg19_bn --pool-num 4  --dataset shanghai-tech-a

python train.py --save-dir /mnt/hdd02/res-bagnet/layer-last/scratch --exp re_qnrf-vgg19-scratch --arch vgg19 --pool-num 4  --dataset rescale-ucf-qnrf
python train.py --save-dir /mnt/hdd02/res-bagnet/layer-last/scratch --exp re_qnrf-vgg19_bn-scratch --arch vgg19_bn --pool-num 4  --dataset rescale-ucf-qnrf
python train.py --save-dir /mnt/hdd02/res-bagnet/layer-last/scratch --exp re_qnrf-bagnet33-scratch --arch bagnet33 --pool-num 5  --dataset rescale-ucf-qnrf
python train.py --save-dir /mnt/hdd02/res-bagnet/layer-last/scratch --exp re_qnrf-resnet50-scratch --arch resnet50 --pool-num 5  --dataset rescale-ucf-qnrf
# モデルでかすぎてVal通らない python train.py --save-dir /mnt/hdd02/res-bagnet/layer-last/scratch --exp re_qnrf-fusionnet-scratch --arch fusionnet --pool-num 5  --dataset rescale-ucf-qnrf

# test
python test.py --save-dir /mnt/hdd02/res-bagnet/layer-last/scratch/stb-bagnet33-scratch --arch bagnet33 --pool-num 5 --dataset shanghai-tech-b
python test.py --save-dir /mnt/hdd02/res-bagnet/layer-last/scratch/stb-resnet50-scratch --arch resnet50 --pool-num 5 --dataset shanghai-tech-b
python test.py --save-dir /mnt/hdd02/res-bagnet/layer-last/scratch/stb-vgg19-scratch --arch vgg19 --pool-num 4 --dataset shanghai-tech-b
python test.py --save-dir /mnt/hdd02/res-bagnet/layer-last/scratch/stb-vgg19_bn-scratch --arch vgg19_bn --pool-num 4 --dataset shanghai-tech-b

python test.py --save-dir /mnt/hdd02/res-bagnet/layer-last/scratch/sta-bagnet33-scratch --arch bagnet33 --pool-num 5 --dataset shanghai-tech-a
python test.py --save-dir /mnt/hdd02/res-bagnet/layer-last/scratch/sta-resnet50-scratch --arch resnet50 --pool-num 5 --dataset shanghai-tech-a
python test.py --save-dir /mnt/hdd02/res-bagnet/layer-last/scratch/sta-vgg19-scratch --arch vgg19 --pool-num 4 --dataset shanghai-tech-a
python test.py --save-dir /mnt/hdd02/res-bagnet/layer-last/scratch/sta-vgg19_bn-scratch --arch vgg19_bn --pool-num 4 --dataset shanghai-tech-a

python test.py --save-dir /mnt/hdd02/res-bagnet/layer-last/scratch/re_qnrf-vgg19-scratch --arch vgg19 --pool-num 4 --dataset rescale-ucf-qnrf
python test.py --save-dir /mnt/hdd02/res-bagnet/layer-last/scratch/re_qnrf-vgg19_bn-scratch --arch vgg19_bn --pool-num 4 --dataset rescale-ucf-qnrf
python test.py --save-dir /mnt/hdd02/res-bagnet/layer-last/scratch/re_qnrf-bagnet33-scratch --arch bagnet33 --pool-num 5 --dataset rescale-ucf-qnrf
python test.py --save-dir /mnt/hdd02/res-bagnet/layer-last/scratch/re_qnrf-resnet50-scratch --arch resnet50 --pool-num 5 --dataset rescale-ucf-qnrf
# モデルでかすぎてVal通らない python test.py --save-dir /mnt/hdd02/res-bagnet/layer-last/scratch/re_qnrf-fusionnet-scratch --arch fusionnet --pool-num 5 --dataset rescale-ucf-qnrf