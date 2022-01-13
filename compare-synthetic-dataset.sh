#!bin/bash

# synthetic-dataset-2d scratch
python train.py --save-dir /mnt/hdd02/res-bagnet/synthetic-2d --exp syn-resnet50-scratch --arch resnet50 --pool-num 5 --dataset synthetic-dataset-2d
python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic-2d/syn-resnet50-scratch --arch resnet50 --pool-num 5 --dataset synthetic-dataset-2d

# synthetic-dataset-2d IMNet pretrained
python train.py --save-dir /mnt/hdd02/res-bagnet/synthetic-2d --exp syn-resnet50-pretrained --arch resnet50 --pool-num 5 --dataset synthetic-dataset-2d --pretrained
python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic-2d/syn-resnet50-pretrained --arch resnet50 --pool-num 5 --dataset synthetic-dataset-2d


# synthetic-dataset-2d-bg scratch
python train.py --save-dir /mnt/hdd02/res-bagnet/synthetic-2d-bg --exp syn-resnet50-scratch --arch resnet50 --pool-num 5 --dataset synthetic-dataset-2d-bg
python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic-2d-bg/syn-resnet50-scratch --arch resnet50 --pool-num 5 --dataset synthetic-dataset-2d-bg

# synthetic-dataset-2d-bg IMNet pretrained
python train.py --save-dir /mnt/hdd02/res-bagnet/synthetic-2d-bg --exp syn-resnet50-pretrained --arch resnet50 --pool-num 5 --dataset synthetic-dataset-2d-bg --pretrained
python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic-2d-bg/syn-resnet50-pretrained --arch resnet50 --pool-num 5 --dataset synthetic-dataset-2d-bg


# synthetic-dataset IMNet pretrained
# python train.py --save-dir /mnt/hdd02/res-bagnet/synthetic --exp syn-resnet50-pretrained --arch resnet50 --pool-num 5 --dataset synthetic-dataset --pretrained
# python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic/syn-resnet50-pretrained --arch resnet50 --pool-num 5 --dataset synthetic-dataset


# synthetic-dataset-v2 IMNet pretrained
# python train.py --save-dir /mnt/hdd02/res-bagnet/synthetic-v2 --exp syn-resnet50-pretrained --arch resnet50 --pool-num 5 --dataset synthetic-dataset-v2 --pretrained
# python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic-v2/syn-resnet50-pretrained --arch resnet50 --pool-num 5 --dataset synthetic-dataset-v2