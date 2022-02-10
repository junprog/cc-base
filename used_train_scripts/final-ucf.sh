#!bin/bash

# データ数比較
### synthetic-1000-scratch ###
# rescale-ucf-qnrf
python train.py --save-dir /mnt/hdd02/res-syn/syn-1000/transfer --exp re_qnrf-resnet50-transfer-scratch --arch resnet50 --pool-num 5 --dataset rescale-ucf-qnrf --resume /mnt/hdd02/res-syn/syn-1000/syn-resnet50-scratch/best_model.pth --lr 1e-5 --max-epoch 401
python test.py --save-dir /mnt/hdd02/res-syn/syn-1000/transfer/re_qnrf-resnet50-transfer-scratch --arch resnet50 --pool-num 5 --dataset rescale-ucf-qnrf

### synthetic-1000-pretrained ###
# rescale-ucf-qnrf
python train.py --save-dir /mnt/hdd02/res-syn/syn-1000/transfer --exp re_qnrf-resnet50-transfer-pretrained --arch resnet50 --pool-num 5 --dataset rescale-ucf-qnrf --resume /mnt/hdd02/res-syn/syn-1000/syn-resnet50-pretrained/best_model.pth --lr 1e-5 --max-epoch 401
python test.py --save-dir /mnt/hdd02/res-syn/syn-1000/transfer/re_qnrf-resnet50-transfer-pretrained --arch resnet50 --pool-num 5 --dataset rescale-ucf-qnrf

### synthetic-2000-scratch ###
# rescale-ucf-qnrf
python train.py --save-dir /mnt/hdd02/res-syn/syn-2000/transfer --exp re_qnrf-resnet50-transfer-scratch --arch resnet50 --pool-num 5 --dataset rescale-ucf-qnrf --resume /mnt/hdd02/res-syn/syn-2000/syn-resnet50-scratch/best_model.pth --lr 1e-5 --max-epoch 401
python test.py --save-dir /mnt/hdd02/res-syn/syn-2000/transfer/re_qnrf-resnet50-transfer-scratch --arch resnet50 --pool-num 5 --dataset rescale-ucf-qnrf

rem ### synthetic-2000-pretrained ###
# rescale-ucf-qnrf
python train.py --save-dir /mnt/hdd02/res-syn/syn-2000/transfer --exp re_qnrf-resnet50-transfer-pretrained --arch resnet50 --pool-num 5 --dataset rescale-ucf-qnrf --resume /mnt/hdd02/res-syn/syn-2000/syn-resnet50-pretrained/best_model.pth --lr 1e-5 --max-epoch 401
python test.py --save-dir /mnt/hdd02/res-syn/syn-2000/transfer/re_qnrf-resnet50-transfer-pretrained --arch resnet50 --pool-num 5 --dataset rescale-ucf-qnrf

### synthetic-4000-scratch ###
# rescale-ucf-qnrf
python train.py --save-dir /mnt/hdd02/res-syn/syn-4000/transfer --exp re_qnrf-resnet50-transfer-scratch --arch resnet50 --pool-num 5 --dataset rescale-ucf-qnrf --resume /mnt/hdd02/res-syn/syn-4000/syn-resnet50-scratch/best_model.pth --lr 1e-5 --max-epoch 401
python test.py --save-dir /mnt/hdd02/res-syn/syn-4000/transfer/re_qnrf-resnet50-transfer-scratch --arch resnet50 --pool-num 5 --dataset rescale-ucf-qnrf

### synthetic-4000-pretrained ###
# rescale-ucf-qnrf
python train.py --save-dir /mnt/hdd02/res-syn/syn-4000/transfer --exp re_qnrf-resnet50-transfer-pretrained --arch resnet50 --pool-num 5 --dataset rescale-ucf-qnrf --resume /mnt/hdd02/res-syn/syn-4000/syn-resnet50-pretrained/best_model.pth --lr 1e-5 --max-epoch 401
python test.py --save-dir /mnt/hdd02/res-syn/syn-4000/transfer/re_qnrf-resnet50-transfer-pretrained --arch resnet50 --pool-num 5 --dataset rescale-ucf-qnrf

# 400 epoch 追試
python train.py --save-dir /mnt/hdd02/res-bagnet/layer-last/imagenet-pretrained --exp re_qnrf-resnet50-pretrained-400ep --arch resnet50 --pool-num 5 --pretrained --dataset rescale-ucf-qnrf
python test.py --save-dir /mnt/hdd02/res-bagnet/layer-last/imagenet-pretrained/re_qnrf-resnet50-pretrained-400ep --arch resnet50 --pool-num 5 --dataset rescale-ucf-qnrf