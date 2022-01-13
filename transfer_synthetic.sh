#!bin/bash

# ↓↓↓↓ from synthetic ↓↓↓↓

# sta
python train.py --save-dir /mnt/hdd02/res-bagnet/synthetic/transfer --exp sta-bagnet33-transfer --arch bagnet33 --pool-num 5 --dataset shanghai-tech-a --resume /mnt/hdd02/res-bagnet/synthetic/syn-bagnet33-scratch/best_model.pth --lr 1e-5 --max-epoch 401
python train.py --save-dir /mnt/hdd02/res-bagnet/synthetic/transfer --exp sta-resnet50-transfer --arch resnet50 --pool-num 5 --dataset shanghai-tech-a --resume /mnt/hdd02/res-bagnet/synthetic/syn-resnet50-scratch/best_model.pth --lr 1e-5 --max-epoch 401
python train.py --save-dir /mnt/hdd02/res-bagnet/synthetic/transfer --exp sta-vgg19-transfer --arch vgg19 --pool-num 4 --dataset shanghai-tech-a --resume /mnt/hdd02/res-bagnet/synthetic/syn-vgg19-scratch/best_model.pth --lr 1e-5 --max-epoch 401
python train.py --save-dir /mnt/hdd02/res-bagnet/synthetic/transfer --exp sta-vgg19_bn-transfer --arch vgg19_bn --pool-num 4 --dataset shanghai-tech-a --resume /mnt/hdd02/res-bagnet/synthetic/syn-vgg19_bn-scratch/best_model.pth --lr 1e-5 --max-epoch 401
#未対応 python train.py --save-dir /mnt/hdd02/res-bagnet/synthetic/transfer --exp sta-fusionnet-transfer --arch fusionnet --pool-num 5 --dataset shanghai-tech-a

# stb
python train.py --save-dir /mnt/hdd02/res-bagnet/synthetic/transfer --exp stb-bagnet33-transfer --arch bagnet33 --pool-num 5 --dataset shanghai-tech-b --resume /mnt/hdd02/res-bagnet/synthetic/syn-bagnet33-scratch/best_model.pth --lr 1e-5 --max-epoch 401 --batch-size 2
python train.py --save-dir /mnt/hdd02/res-bagnet/synthetic/transfer --exp stb-resnet50-transfer --arch resnet50 --pool-num 5 --dataset shanghai-tech-b --resume /mnt/hdd02/res-bagnet/synthetic/syn-resnet50-scratch/best_model.pth --lr 1e-5 --max-epoch 401 --batch-size 4
python train.py --save-dir /mnt/hdd02/res-bagnet/synthetic/transfer --exp stb-vgg19-transfer --arch vgg19 --pool-num 4 --dataset shanghai-tech-b --resume /mnt/hdd02/res-bagnet/synthetic/syn-vgg19-scratch/best_model.pth --lr 1e-5 --max-epoch 401
python train.py --save-dir /mnt/hdd02/res-bagnet/synthetic/transfer --exp stb-vgg19_bn-transfer --arch vgg19_bn --pool-num 4 --dataset shanghai-tech-b --resume /mnt/hdd02/res-bagnet/synthetic/syn-vgg19_bn-scratch/best_model.pth --lr 1e-5 --max-epoch 401

# rescale-ucf-qnrf
python train.py --save-dir /mnt/hdd02/res-bagnet/synthetic/transfer --exp re_qnrf-bagnet33-transfer --arch bagnet33 --pool-num 5 --dataset rescale-ucf-qnrf --resume /mnt/hdd02/res-bagnet/synthetic/syn-bagnet33-scratch/best_model.pth --lr 1e-5 --max-epoch 401
python train.py --save-dir /mnt/hdd02/res-bagnet/synthetic/transfer --exp re_qnrf-resnet50-transfer --arch resnet50 --pool-num 5 --dataset rescale-ucf-qnrf --resume /mnt/hdd02/res-bagnet/synthetic/syn-resnet50-scratch/best_model.pth --lr 1e-5 --max-epoch 401
python train.py --save-dir /mnt/hdd02/res-bagnet/synthetic/transfer --exp re_qnrf-vgg19-transfer --arch vgg19 --pool-num 4 --dataset rescale-ucf-qnrf --resume /mnt/hdd02/res-bagnet/synthetic/syn-vgg19-scratch/best_model.pth --lr 1e-5 --max-epoch 401
python train.py --save-dir /mnt/hdd02/res-bagnet/synthetic/transfer --exp re_qnrf-vgg19_bn-transfer --arch vgg19_bn --pool-num 4 --dataset rescale-ucf-qnrf --resume /mnt/hdd02/res-bagnet/synthetic/syn-vgg19_bn-scratch/best_model.pth --lr 1e-5 --max-epoch 401

# sta
python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic/transfer/sta-bagnet33-transfer --arch bagnet33 --pool-num 5 --dataset shanghai-tech-a
python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic/transfer/sta-resnet50-transfer --arch resnet50 --pool-num 5 --dataset shanghai-tech-a
python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic/transfer/sta-vgg19-transfer --arch vgg19 --pool-num 4 --dataset shanghai-tech-a
python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic/transfer/sta-vgg19_bn-transfer --arch vgg19_bn --pool-num 4 --dataset shanghai-tech-a
#python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic/transfer/sta-fusionnet-transfer  --arch fusionnet --pool-num 5 --dataset shanghai-tech-a

# stb
python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic/transfer/stb-bagnet33-transfer --arch bagnet33 --pool-num 5 --dataset shanghai-tech-b
python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic/transfer/stb-resnet50-transfer --arch resnet50 --pool-num 5 --dataset shanghai-tech-b
python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic/transfer/stb-vgg19-transfer --arch vgg19 --pool-num 4 --dataset shanghai-tech-b
python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic/transfer/stb-vgg19_bn-transfer --arch vgg19_bn --pool-num 4 --dataset shanghai-tech-b
#python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic/transfer/stb-fusionnet-transfer  --arch fusionnet --pool-num 5 --dataset shanghai-tech-b

# rescale-ucf
python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic/transfer/re_qnrf-bagnet33-transfer --arch bagnet33 --pool-num 5 --dataset rescale-ucf-qnrf
python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic/transfer/re_qnrf-resnet50-transfer --arch resnet50 --pool-num 5 --dataset rescale-ucf-qnrf
python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic/transfer/re_qnrf-vgg19-transfer --arch vgg19 --pool-num 4 --dataset rescale-ucf-qnrf
python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic/transfer/re_qnrf-vgg19_bn-transfer --arch vgg19_bn --pool-num 4 --dataset rescale-ucf-qnrf
#python test.py --save-dir /mnt/hdd02/res-bagnet/synthetic/transfer/re_qnrf-fusionnet-transfer  --arch fusionnet --pool-num 5 --dataset rescale-ucf-qnrf