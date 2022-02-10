#!bin/bash

# ### synthetic-v1-pretrained ###
# # sta
# python train.py --save-dir /mnt/hdd02/res-bagnet/synthetic/transfer --exp sta-resnet50-transfer-pretrained --arch resnet50 --pool-num 5 --dataset shanghai-tech-a --resume /mnt/hdd02/res-bagnet/synthetic/syn-resnet50-pretrained/best_model.pth --lr 1e-5 --max-epoch 401
# python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic/transfer/sta-resnet50-transfer-pretrained --arch resnet50 --pool-num 5 --dataset shanghai-tech-a

# # stb
# python train.py --save-dir /mnt/hdd02/res-bagnet/synthetic/transfer --exp stb-resnet50-transfer-pretrained --arch resnet50 --pool-num 5 --dataset shanghai-tech-b --resume /mnt/hdd02/res-bagnet/synthetic/syn-resnet50-pretrained/best_model.pth --lr 1e-5 --max-epoch 401 --batch-size 4
# python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic/transfer/stb-resnet50-transfer-pretrained --arch resnet50 --pool-num 5 --dataset shanghai-tech-b

# # rescale-ucf-qnrf
# python train.py --save-dir /mnt/hdd02/res-bagnet/synthetic/transfer --exp re_qnrf-resnet50-transfer-pretrained --arch resnet50 --pool-num 5 --dataset rescale-ucf-qnrf --resume /mnt/hdd02/res-bagnet/synthetic/syn-resnet50-pretrained/best_model.pth --lr 1e-5 --max-epoch 401
# python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic/transfer/re_qnrf-resnet50-transfer-pretrained --arch resnet50 --pool-num 5 --dataset rescale-ucf-qnrf


# ### synthetic-v2-pretrained ###
# # sta
# python train.py --save-dir /mnt/hdd02/res-bagnet/synthetic-v2/transfer --exp sta-resnet50-transfer-pretrained --arch resnet50 --pool-num 5 --dataset shanghai-tech-a --resume /mnt/hdd02/res-bagnet/synthetic-v2/syn-resnet50-pretrained/best_model.pth --lr 1e-5 --max-epoch 401
# python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic-v2/transfer/sta-resnet50-transfer-pretrained --arch resnet50 --pool-num 5 --dataset shanghai-tech-a

# # stb
# python train.py --save-dir /mnt/hdd02/res-bagnet/synthetic-v2/transfer --exp stb-resnet50-transfer-pretrained --arch resnet50 --pool-num 5 --dataset shanghai-tech-b --resume /mnt/hdd02/res-bagnet/synthetic-v2/syn-resnet50-pretrained/best_model.pth --lr 1e-5 --max-epoch 401 --batch-size 4
# python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic-v2/transfer/stb-resnet50-transfer-pretrained --arch resnet50 --pool-num 5 --dataset shanghai-tech-b

# # rescale-ucf-qnrf
# python train.py --save-dir /mnt/hdd02/res-bagnet/synthetic-v2/transfer --exp re_qnrf-resnet50-transfer-pretrained --arch resnet50 --pool-num 5 --dataset rescale-ucf-qnrf --resume /mnt/hdd02/res-bagnet/synthetic-v2/syn-resnet50-pretrained/best_model.pth --lr 1e-5 --max-epoch 401
# python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic-v2/transfer/re_qnrf-resnet50-transfer-pretrained --arch resnet50 --pool-num 5 --dataset rescale-ucf-qnrf


### synthetic-2d-scratch ###
# sta
python train.py --save-dir /mnt/hdd02/res-bagnet/synthetic-2d/transfer --exp sta-resnet50-transfer-scratch --arch resnet50 --pool-num 5 --dataset shanghai-tech-a --resume /mnt/hdd02/res-bagnet/synthetic-2d/syn-resnet50-scratch/best_model.pth --lr 1e-5 --max-epoch 401
python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic-2d/transfer/sta-resnet50-transfer-scratch --arch resnet50 --pool-num 5 --dataset shanghai-tech-a

# stb
python train.py --save-dir /mnt/hdd02/res-bagnet/synthetic-2d/transfer --exp stb-resnet50-transfer-scratch --arch resnet50 --pool-num 5 --dataset shanghai-tech-b --resume /mnt/hdd02/res-bagnet/synthetic-2d/syn-resnet50-scratch/best_model.pth --lr 1e-5 --max-epoch 401 --batch-size 4
python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic-2d/transfer/stb-resnet50-transfer-scratch --arch resnet50 --pool-num 5 --dataset shanghai-tech-b

# rescale-ucf-qnrf
python train.py --save-dir /mnt/hdd02/res-bagnet/synthetic-2d/transfer --exp re_qnrf-resnet50-transfer-scratch --arch resnet50 --pool-num 5 --dataset rescale-ucf-qnrf --resume /mnt/hdd02/res-bagnet/synthetic-2d/syn-resnet50-scratch/best_model.pth --lr 1e-5 --max-epoch 401
python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic-2d/transfer/re_qnrf-resnet50-transfer-scratch --arch resnet50 --pool-num 5 --dataset rescale-ucf-qnrf

### synthetic-2d-pretrained ###
# sta
python train.py --save-dir /mnt/hdd02/res-bagnet/synthetic-2d/transfer --exp sta-resnet50-transfer-pretrained --arch resnet50 --pool-num 5 --dataset shanghai-tech-a --resume /mnt/hdd02/res-bagnet/synthetic-2d/syn-resnet50-pretrained/best_model.pth --lr 1e-5 --max-epoch 401
python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic-2d/transfer/sta-resnet50-transfer-pretrained --arch resnet50 --pool-num 5 --dataset shanghai-tech-a

# stb
python train.py --save-dir /mnt/hdd02/res-bagnet/synthetic-2d/transfer --exp stb-resnet50-transfer-pretrained --arch resnet50 --pool-num 5 --dataset shanghai-tech-b --resume /mnt/hdd02/res-bagnet/synthetic-2d/syn-resnet50-pretrained/best_model.pth --lr 1e-5 --max-epoch 401 --batch-size 4
python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic-2d/transfer/stb-resnet50-transfer-pretrained --arch resnet50 --pool-num 5 --dataset shanghai-tech-b

# rescale-ucf-qnrf
python train.py --save-dir /mnt/hdd02/res-bagnet/synthetic-2d/transfer --exp re_qnrf-resnet50-transfer-pretrained --arch resnet50 --pool-num 5 --dataset rescale-ucf-qnrf --resume /mnt/hdd02/res-bagnet/synthetic-2d/syn-resnet50-pretrained/best_model.pth --lr 1e-5 --max-epoch 401
python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic-2d/transfer/re_qnrf-resnet50-transfer-pretrained --arch resnet50 --pool-num 5 --dataset rescale-ucf-qnrf


### synthetic-2d-bg-scratch ###
# sta
python train.py --save-dir /mnt/hdd02/res-bagnet/synthetic-2d-bg/transfer --exp sta-resnet50-transfer-scratch --arch resnet50 --pool-num 5 --dataset shanghai-tech-a --resume /mnt/hdd02/res-bagnet/synthetic-2d-bg/syn-resnet50-scratch/best_model.pth --lr 1e-5 --max-epoch 401
python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic-2d-bg/transfer/sta-resnet50-transfer-scratch --arch resnet50 --pool-num 5 --dataset shanghai-tech-a

# stb
python train.py --save-dir /mnt/hdd02/res-bagnet/synthetic-2d-bg/transfer --exp stb-resnet50-transfer-scratch --arch resnet50 --pool-num 5 --dataset shanghai-tech-b --resume /mnt/hdd02/res-bagnet/synthetic-2d-bg/syn-resnet50-scratch/best_model.pth --lr 1e-5 --max-epoch 401 --batch-size 4
python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic-2d-bg/transfer/stb-resnet50-transfer-scratch --arch resnet50 --pool-num 5 --dataset shanghai-tech-b

# rescale-ucf-qnrf
python train.py --save-dir /mnt/hdd02/res-bagnet/synthetic-2d-bg/transfer --exp re_qnrf-resnet50-transfer-scratch --arch resnet50 --pool-num 5 --dataset rescale-ucf-qnrf --resume /mnt/hdd02/res-bagnet/synthetic-2d-bg/syn-resnet50-scratch/best_model.pth --lr 1e-5 --max-epoch 401
python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic-2d-bg/transfer/re_qnrf-resnet50-transfer-scratch --arch resnet50 --pool-num 5 --dataset rescale-ucf-qnrf


### synthetic-2d-bg-pretrained ###
# sta
python train.py --save-dir /mnt/hdd02/res-bagnet/synthetic-2d-bg/transfer --exp sta-resnet50-transfer-pretrained --arch resnet50 --pool-num 5 --dataset shanghai-tech-a --resume /mnt/hdd02/res-bagnet/synthetic-2d-bg/syn-resnet50-pretrained/best_model.pth --lr 1e-5 --max-epoch 401
python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic-2d-bg/transfer/sta-resnet50-transfer-pretrained --arch resnet50 --pool-num 5 --dataset shanghai-tech-a

# stb
python train.py --save-dir /mnt/hdd02/res-bagnet/synthetic-2d-bg/transfer --exp stb-resnet50-transfer-pretrained --arch resnet50 --pool-num 5 --dataset shanghai-tech-b --resume /mnt/hdd02/res-bagnet/synthetic-2d-bg/syn-resnet50-pretrained/best_model.pth --lr 1e-5 --max-epoch 401 --batch-size 4
python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic-2d-bg/transfer/stb-resnet50-transfer-pretrained --arch resnet50 --pool-num 5 --dataset shanghai-tech-b

# rescale-ucf-qnrf
python train.py --save-dir /mnt/hdd02/res-bagnet/synthetic-2d-bg/transfer --exp re_qnrf-resnet50-transfer-pretrained --arch resnet50 --pool-num 5 --dataset rescale-ucf-qnrf --resume /mnt/hdd02/res-bagnet/synthetic-2d-bg/syn-resnet50-pretrained/best_model.pth --lr 1e-5 --max-epoch 401
python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic-2d-bg/transfer/re_qnrf-resnet50-transfer-pretrained --arch resnet50 --pool-num 5 --dataset rescale-ucf-qnrf