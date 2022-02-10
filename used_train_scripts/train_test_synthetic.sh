#!bin/bash

# train
python train.py --save-dir /mnt/hdd02/res-bagnet/layer-last --exp syn-bagnet33-scratch --arch bagnet33 --pool-num 5 --dataset synthetic-dataset
python train.py --save-dir /mnt/hdd02/res-bagnet/layer-last --exp syn-resnet50-scratch --arch resnet50 --pool-num 5 --dataset synthetic-dataset
python train.py --save-dir /mnt/hdd02/res-bagnet/layer-last --exp syn-vgg19-scratch --arch vgg19 --pool-num 4 --dataset synthetic-dataset
python train.py --save-dir /mnt/hdd02/res-bagnet/layer-last --exp syn-vgg19_bn-scratch --arch vgg19_bn --pool-num 4 --dataset synthetic-dataset
python train.py --save-dir /mnt/hdd02/res-bagnet/layer-last --exp syn-fusionnet-scratch --arch fusionnet --pool-num 5 --dataset synthetic-dataset

# test
python test.py --save-dir /mnt/hdd02/res-bagnet/layer-last/syn-bagnet33-scratch --arch bagnet33 --pool-num 5 --dataset synthetic-dataset
python test.py --save-dir /mnt/hdd02/res-bagnet/layer-last/syn-resnet50-scratch --arch resnet50 --pool-num 5 --dataset synthetic-dataset
python test.py --save-dir /mnt/hdd02/res-bagnet/layer-last/syn-vgg19-scratch --arch vgg19 --pool-num 4 --dataset synthetic-dataset
python test.py --save-dir /mnt/hdd02/res-bagnet/layer-last/syn-vgg19_bn-scratch --arch vgg19_bn --pool-num 4 --dataset synthetic-dataset
python test.py --save-dir /mnt/hdd02/res-bagnet/layer-last/syn-fusionnet-scratch  --arch fusionnet --pool-num 5 --dataset synthetic-dataset