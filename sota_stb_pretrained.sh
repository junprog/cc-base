#!bin/bash

#### 実行前に fusion_net.py の feet_freeze 部分書き換え！！！！ ###

# 流れ
# synthetic-2d を使用
# 1. ResNet: IN - SYN - TAR 済(/mnt/hdd02/res-bagnet/synthetic-2d/transfer/stb-resnet50-transfer-pretrained)
# 2. BagNet: IN - SYN - TAR 未,未
# 3. fusion(Res, Bag freeze) 未

# BagNet IN - SYN
# python train.py --save-dir /mnt/hdd02/res-bagnet/synthetic-2d --exp syn-bagnet33-pretrained --arch bagnet33 --pool-num 5 --dataset synthetic-dataset-2d --pretrained
# python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic-2d/syn-bagnet33-pretrained --arch bagnet33 --pool-num 5 --dataset synthetic-dataset-2d

# BagNet IN - SYN - TAR
python train.py --save-dir /mnt/hdd02/res-bagnet/synthetic-2d/transfer --exp stb-bagnet33-transfer-pretrained --arch bagnet33 --pool-num 5 --dataset shanghai-tech-b --resume /mnt/hdd02/res-bagnet/synthetic-2d/syn-bagnet33-pretrained/best_model.pth --batch-size 2
python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic-2d/transfer/stb-bagnet33-transfer-pretrained --arch bagnet33 --pool-num 5 --dataset shanghai-tech-b

# fusion (Res, Bag freeze)
python train.py --save-dir /mnt/hdd02/res-bagnet/sota --exp stb-fusionnet-freeze-pretrained --arch fusionnet --pool-num 5 --feat-freeze --dataset shanghai-tech-b --batch-size 2
python test.py --save-dir /mnt/hdd02/res-bagnet/sota/stb-fusionnet-freeze-pretrained --arch fusionnet --pool-num 5 --dataset shanghai-tech-b