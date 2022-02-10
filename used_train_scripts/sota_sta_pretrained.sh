#!bin/bash

# 流れ
# synthetic-3d-bg を使用
# 1. ResNet: IN - SYN - TAR 済(/mnt/hdd02/res-bagnet/synthetic-v2/transfer/sta-resnet50-transfer-pretrained)
# 2. BagNet: IN - SYN - TAR 未,未
# 3. fusion(Res, Bag freeze) 未

# BagNet IN - SYN
# python train.py --save-dir /mnt/hdd02/res-bagnet/synthetic-v2 --exp syn-bagnet33-pretrained --arch bagnet33 --pool-num 5 --dataset synthetic-dataset-v2 --pretrained
# python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic-v2/syn-bagnet33-pretrained --arch bagnet33 --pool-num 5 --dataset synthetic-dataset-v2

# BagNet IN - SYN - TAR
# python train.py --save-dir /mnt/hdd02/res-bagnet/synthetic-v2/transfer --exp sta-bagnet33-transfer-pretrained --arch bagnet33 --pool-num 5 --dataset shanghai-tech-a --resume /mnt/hdd02/res-bagnet/synthetic-v2/syn-bagnet33-pretrained/best_model.pth --batch-size 2
# python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic-v2/transfer/sta-bagnet33-transfer-pretrained --arch bagnet33 --pool-num 5 --dataset shanghai-tech-a

# fusion (Res, Bag freeze)
python train.py --save-dir /mnt/hdd02/res-bagnet/sota --exp sta-fusionnet-freeze-pretrained --arch fusionnet --pool-num 5 --feat-freeze --dataset shanghai-tech-a --batch-size 4
python test.py --save-dir /mnt/hdd02/res-bagnet/sota/sta-fusionnet-freeze-pretrained --arch fusionnet --pool-num 5 --dataset shanghai-tech-a