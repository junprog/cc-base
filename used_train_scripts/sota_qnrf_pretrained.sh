#!bin/bash

# 流れ
# synthetic-2d-bg を使用
# 1. ResNet: IN - SYN - TAR 済(/mnt/hdd02/res-bagnet/synthetic-2d-bg/transfer/re_qnrf-resnet50-transfer-pretrained)
# 2. BagNet: IN - SYN - TAR 未,未
# 3. fusion(Res, Bag freeze) 未

# BagNet IN - SYN
# python train.py --save-dir /mnt/hdd02/res-bagnet/synthetic-2d-bg --exp syn-bagnet33-pretrained --arch bagnet33 --pool-num 5 --dataset synthetic-dataset-2d-bg --pretrained
# python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic-2d-bg/syn-bagnet33-pretrained --arch bagnet33 --pool-num 5 --dataset synthetic-dataset-2d-bg

# BagNet IN - SYN - TAR
# python train.py --save-dir /mnt/hdd02/res-bagnet/synthetic-2d-bg/transfer --exp re_qnrf-bagnet33-transfer-pretrained --arch bagnet33 --pool-num 5 --dataset rescale-ucf-qnrf --resume /mnt/hdd02/res-bagnet/synthetic-2d-bg/syn-bagnet33-pretrained/best_model.pth --batch-size 2
# python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic-2d-bg/transfer/re_qnrf-bagnet33-transfer-pretrained --arch bagnet33 --pool-num 5 --dataset rescale-ucf-qnrf

# fusion (Res, Bag freeze)
python train.py --save-dir /mnt/hdd02/res-bagnet/sota --exp re_qnrf-fusionnet-freeze-pretrained --arch fusionnet --pool-num 5 --feat-freeze --dataset rescale-ucf-qnrf --batch-size 4
python test.py --save-dir /mnt/hdd02/res-bagnet/sota/re_qnrf-fusionnet-freeze-pretrained --arch fusionnet --pool-num 5 --dataset rescale-ucf-qnrf