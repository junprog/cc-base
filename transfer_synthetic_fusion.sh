#!bin/bash

# ↓↓↓↓ from synthetic ↓↓↓↓

# sta
python train.py --save-dir /mnt/hdd02/res-bagnet/synthetic/transfer --exp sta-fusionnet-transfer --arch fusionnet --pool-num 5 --dataset shanghai-tech-a --resume /mnt/hdd02/res-bagnet/synthetic/syn-fusionnet-scratch/best_model.pth --lr 1e-5 --max-epoch 401 --batch-size 2

# stb
python train.py --save-dir /mnt/hdd02/res-bagnet/synthetic/transfer --exp stb-fusionnet-transfer --arch fusionnet --pool-num 5 --dataset shanghai-tech-b --resume /mnt/hdd02/res-bagnet/synthetic/syn-fusionnet-scratch/best_model.pth --lr 1e-5 --max-epoch 401 --batch-size 1

# rescale-ucf-qnrf
python train.py --save-dir /mnt/hdd02/res-bagnet/synthetic/transfer --exp re_qnrf-fusionnet-transfer --arch fusionnet --pool-num 5 --dataset rescale-ucf-qnrf --resume /mnt/hdd02/res-bagnet/synthetic/syn-fusionnet-scratch/best_model.pth --lr 1e-5 --max-epoch 401  --batch-size 2

# sta
python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic/transfer/sta-fusionnet-transfer  --arch fusionnet --pool-num 5 --dataset shanghai-tech-a

# stb
python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic/transfer/stb-fusionnet-transfer  --arch fusionnet --pool-num 5 --dataset shanghai-tech-b

# rescale-ucf
python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic/transfer/re_qnrf-fusionnet-transfer  --arch fusionnet --pool-num 5 --dataset rescale-ucf-qnrf