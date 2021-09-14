#!bin/bash

# train
#python train.py --save-dir /mnt/hdd02/res-rgbd --exp exp01-1 --arch vgg19_bn --pool-num 3 --up-scale 1 --rgb
#python train.py --save-dir /mnt/hdd02/res-rgbd --exp exp01-2 --arch resnet50 --pool-num 4 --up-scale 1 --rgb
python train.py --save-dir /mnt/hdd02/res-rgbd --exp exp01-3 --arch csrnet --pool-num 3 --up-scale 8 --rgb --lr 1e-5
#python train.py --save-dir /mnt/hdd02/res-rgbd --exp exp01-4 --arch mcnn --pool-num 2 --up-scale 1 --rgb

python train.py --save-dir /mnt/hdd02/res-rgbd --exp exp02-1 --arch vgg19_bn --pool-num 3 --up-scale 1 --depth
#python train.py --save-dir /mnt/hdd02/res-rgbd --exp exp02-2 --arch resnet50 --pool-num 4 --up-scale 1 --depth
python train.py --save-dir /mnt/hdd02/res-rgbd --exp exp02-3 --arch csrnet --pool-num 3 --up-scale 1 --depth --lr 1e-6
#python train.py --save-dir /mnt/hdd02/res-rgbd --exp exp02-4 --arch mcnn --pool-num 2 --up-scale 1 --depth

python train.py --save-dir /mnt/hdd02/res-rgbd --exp exp03-1 --arch vgg19_bn --pool-num 3 --up-scale 1 --depth --rgb
#python train.py --save-dir /mnt/hdd02/res-rgbd --exp exp03-2 --arch resnet50 --pool-num 4 --up-scale 1 --depth --rgb
python train.py --save-dir /mnt/hdd02/res-rgbd --exp exp03-3 --arch csrnet --pool-num 3 --up-scale 1 --depth --rgb --lr 1e-6
#python train.py --save-dir /mnt/hdd02/res-rgbd --exp exp03-4 --arch mcnn --pool-num 2 --up-scale 1 --depth --rgb

# test
python test.py --save-dir /mnt/hdd02/res-rgbd/exp01-1 --arch vgg19_bn --pool-num 3 --up-scale 1 --rgb
#python test.py --save-dir /mnt/hdd02/res-rgbd/exp01-2 --arch resnet50 --pool-num 4 --up-scale 1 --rgb
python test.py --save-dir /mnt/hdd02/res-rgbd/exp01-3 --arch csrnet --pool-num 3 --up-scale 1 --rgb 
#python test.py --save-dir /mnt/hdd02/res-rgbd/exp01-4 --arch mcnn --pool-num 2 --up-scale 1 --rgb

python test.py --save-dir /mnt/hdd02/res-rgbd/exp02-1 --arch vgg19_bn --pool-num 3 --up-scale 1 --depth
#python test.py --save-dir /mnt/hdd02/res-rgbd/exp02-2 --arch resnet50 --pool-num 4 --up-scale 1 --depth
python test.py --save-dir /mnt/hdd02/res-rgbd/exp02-3 --arch csrnet --pool-num 3 --up-scale 1 --depth
#python test.py --save-dir /mnt/hdd02/res-rgbd/exp02-4 --arch mcnn --pool-num 2 --up-scale 1 --depth

python test.py --save-dir /mnt/hdd02/res-rgbd/exp03-1 --arch vgg19_bn --pool-num 3 --up-scale 1 --depth --rgb
#python test.py --save-dir /mnt/hdd02/res-rgbd/exp03-2 --arch resnet50 --pool-num 4 --up-scale 1 --depth --rgb
python test.py --save-dir /mnt/hdd02/res-rgbd/exp03-3 --arch csrnet --pool-num 3 --up-scale 1 --depth --rgb
#python test.py --save-dir /mnt/hdd02/res-rgbd/exp03-4 --arch mcnn --pool-num 2 --up-scale 1 --depth --rgb
